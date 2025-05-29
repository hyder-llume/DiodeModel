#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
from scipy.special import lambertw

def calculate_VT(T):
    # Boltzmann constant in eV/K
    k = 8.617333262145e-5
    return k * T

def func(VS, IS, N, RS, RP, VG, T=300):
    # VS: Applied voltage
    # IS: Saturation current
    # N: Ideality factor
    # RS: Series resistance
    # RP: Parallel resistance
    # VG: Bandgap voltage (turn-on voltage)
    # T: Temperature in Kelvin
    
    VT = calculate_VT(T)
    
    # Adjust voltage for bandgap
    V_adjusted = VS - VG
    
    # Using Lambert W function to solve the implicit equation with both series and parallel resistance
    w = lambertw((IS * RS /(N * VT)) * np.exp((V_adjusted + IS * RS) / (N * VT)))
    
    # Calculate diode current
    ID = IS * ((w * N * VT / (RS * IS)) - 1)
    
    # Add parallel resistance contribution
    if RP > 0:  # Avoid division by zero
        IP = V_adjusted / RP
        I_total = ID + IP
    else:
        I_total = ID
    
    # Convert to mA and return log for fitting stability
    return np.log10(I_total * 1000).real

def plot(xdata, ydata, nPoints, IS, N, RS, RP, VG, T=300):
    # generate points to plot
    vMin = xdata[0]
    vMax = xdata[len(xdata) - 1]
    vRange = vMax - vMin
    vMin = vMin - 0.1 * vRange
    vMax = vMax + 0.1 * vRange
    vStep = (vMax - vMin) / (nPoints - 1)
    VS = []
    ID = []
    for i in range(0, nPoints):
        voltage = vMin + i * vStep
        current = func(voltage, IS, N, RS, RP, VG, T)
        VS.append(voltage)
        ID.append(10**current)

    plt.figure().canvas.manager.set_window_title('LED I-V Characteristic')
    plt.semilogy(xdata, ydata, 'r*', VS, ID, 'b-')
    plt.ylabel('Current / mA')
    plt.xlabel('Voltage / V')
    plt.title('LED I-V Characteristic')
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(prog='DiodeModel.py')
    parser.add_argument('filename', help='Name of file containing I-V data (I in mA, V in volts)')
    parser.add_argument('-c', '--convert', help='Convert read in current to mA', action="store_true")
    parser.add_argument('-p', '--plot', help='Just plot the data and initial guess, no fitting performed', action="store_true")
    parser.add_argument('-IS', '--IS', type=float, default=1e-20, help='Initial guess at saturation current (default = 1e-20 A)')
    parser.add_argument('-N', '--N', type=float, default=2.5, help='Initial guess at Emission coefficient (default = 2.5)')
    parser.add_argument('-RS', '--RS', type=float, default=5, help='Initial guess at ohmic resistance (default = 5 ohm)')
    parser.add_argument('-RP', '--RP', type=float, default=1e6, help='Initial guess at parallel resistance (default = 1M ohm)')
    parser.add_argument('-VG', '--VG', type=float, default=2.0, help='Initial guess at bandgap voltage (default = 2.0V)')
    parser.add_argument('-T', '--T', type=float, default=300, help='Operating temperature in Kelvin (default = 300K)')
    parser.add_argument('-m', '--maxit', type=int, default=2000, help='Maximum number of iterations (default = 2000)')
    parser.add_argument('-n', '--npoints', type=int, default=250, help='Number of points in plot (default = 250)')
    parser.add_argument('-s', '--save', help='Save fit parameters to file', action="store_true")
    parser.add_argument('-f', '--fitfile', type=str, default="", nargs='?', help='Load fit parameters from file')
    args = parser.parse_args()
    
#Read in data from file (V in volt, I in milliamp)
    xdata, ydata = np.loadtxt(args.filename, unpack=True)
    if args.convert:           #Data is in A convert to mA
        ydata = ydata * 1000
    logydata = np.log10(ydata) #log of current to produce stable fit

#Set up initial guess
    params = dict(IS = args.IS, N = args.N, RS = args.RS, RP = args.RP, VG = args.VG)

#Check if input file is given
    if args.fitfile is None:                              #Use default filename
        fitFile = args.filename.split('.')[0] + ".fit"
        try:
            fh = open(fitFile, 'r')
            for line in fh:
                name = line.split('=')[0].strip()
                value = float(line.split('=')[1].strip())
                params[name] = value
            fh.close()
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
    elif len(args.fitfile) > 0:                           #Use input filename
        fitFile = args.fitfile
        try:
            fh = open(fitFile, 'r')
            for line in fh:
                name = line.split('=')[0].strip()
                value = float(line.split('=')[1].strip())
                params[name] = value
            fh.close()
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
    
    if args.plot:
        try:
#Plot data and initial guess
            print("Plotting characteristic with following parameters" + 
                "\nIS = " + str(params['IS']) + "\nN = " + str(params['N']) + "\nRS = " + str(params['RS']) + 
                "\nRP = " + str(params['RP']) + "\nVG = " + str(params['VG']) + "\nT = " + str(args.T))
            plot(xdata, ydata, args.npoints, params['IS'], params['N'], params['RS'], params['RP'], params['VG'], args.T)
        except:
            print("Plotting error")
            exit()
    else:
#Perform non-linear least squares fit
        try:
            popt, pcov, infodict, errmsg, ier = curve_fit(
                lambda x, IS, N, RS, RP, VG: func(x, IS, N, RS, RP, VG, args.T), 
                xdata, logydata, 
                p0=(params['IS'], params['N'], params['RS'], params['RP'], params['VG']), 
                maxfev=args.maxit, 
                full_output=True)
            
#Print converged fit parameters
            print("Fit converged in " + str(infodict['nfev']) + " iterations with the following parameters")
            printString = (f"IS = {popt[0]:.2e} A\n"
                         f"N = {popt[1]:.3f}\n"
                         f"RS = {popt[2]:.3f} Ω\n"
                         f"RP = {popt[3]:.2e} Ω\n"
                         f"VG = {popt[4]:.3f} V\n"
                         f"T = {args.T:.1f} K")
            print(printString)

#Output parameters to file
            if args.save:
                print("\nWriting fit parameters to file")
                outFile = args.filename.split('.')[0] + ".fit"
                try:
                    fh = open(outFile,"w")
                    fh.write(printString)
                    fh.close()
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))

#generate a plot of the fit
            plot(xdata, ydata, args.npoints, popt[0], popt[1], popt[2], popt[3], popt[4], args.T)

        except RuntimeError:
            print("Error - Fit did not converge, try adjusting the starting guess")
            print(errmsg)

if __name__ == "__main__":
    main()
