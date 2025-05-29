#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
from scipy.special import lambertw

VT = 26e-3  # Thermal voltage

# --- FITTING TARGET FUNCTION ---
# def diode_model(VS, IS, N, RS, return_log=True):
#     VS = np.asarray(VS)
#     arg = (IS * RS / (N * VT)) * np.exp((VS + IS * RS) / (N * VT))
#     w = lambertw(arg)
#     I = (N * VT / RS) * w.real - IS
#     I[I <= 0] = 1e-30  # Clamp to avoid log10 of nonpositive values
#     if return_log:
#         return np.log10(I)  # mA, log10
#     else:
#         return I  # mA, linear

def diode_model(V, IS, N, RS, return_log=True):
    V = np.asarray(V)
    VT = 26e-3

    def solve_current(v):
        # Vectorized fixed-point iteration: I = IS * (exp((v - I*RS)/(N*VT)) - 1)
        I = np.full_like(v, 1e-9)  # Initial guess: small forward current
        for _ in range(100):
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                exp_term = np.exp((v - I * RS) / (N * VT))
                I_new = IS * (exp_term - 1)
                I_new = np.where(np.isnan(I_new) | (I_new <= 0), 1e-30, I_new)
            if np.allclose(I, I_new, rtol=1e-4):
                break
            I = I_new
        return I

    Iout = solve_current(V)
    Iout = np.where(Iout <= 0, 1e-30, Iout)  # Safe clamp
    return np.log10(Iout)


# --- PLOT ---
def plot(xdata, ydata, nPoints, IS, N, RS, return_log=True):
    vMin, vMax = xdata[0], xdata[-1]
    VS = np.linspace(vMin, vMax, nPoints)
    
    # Use log-mode always for consistent plotting
    ID_log = diode_model(VS, IS, N, RS, return_log=True)
    ID = 10 ** ID_log  * 1000# Convert log10(mA) → mA

    plt.figure().canvas.manager.set_window_title('Plot Window')
    plt.semilogy(xdata, ydata, 'r*', label="Data")
    plt.semilogy(VS, ID*1000, 'b-', label="Model")
    plt.ylabel('Current (mA)')
    plt.xlabel('Voltage (V)')
    plt.title('Diode I-V Characteristic')
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(prog='DiodeModel.py')
    parser.add_argument('filename', help='I-V data file (V, I in mA or A)')
    parser.add_argument('-c', '--convert', action="store_true", help='Convert current to mA')
    parser.add_argument('-p', '--plot', action="store_true", help='Plot only, no fitting')
    parser.add_argument('-IS', '--IS', type=float, default=1e-14)
    parser.add_argument('-N', '--N', type=float, default=1.5)
    parser.add_argument('-RS', '--RS', type=float, default=10)
    parser.add_argument('-m', '--maxit', type=int, default=20000)
    parser.add_argument('-n', '--npoints', type=int, default=250)
    parser.add_argument('-s', '--save', action="store_true", help='Save fit to .fit file')
    parser.add_argument('-f', '--fitfile', type=str, default="", nargs='?')
    parser.add_argument('--linear', action='store_true', help='Fit in current space instead of log10')
    parser.add_argument('--vcut', type=float, default=1.75, help='Voltage cutoff for turn-on region')
    args = parser.parse_args()

    xdata, ydata = np.loadtxt(args.filename, unpack=True)
    if args.convert:
        ydata *= 1000  # Convert A → mA

    # Mask data below turn-on threshold and <= 0
    mask = (xdata > args.vcut) & (ydata > 0)
    xdata, ydata = xdata[mask], ydata[mask]
    ytarget = np.log10(ydata / 1000) if not args.linear else ydata

    params = {'IS': args.IS, 'N': args.N, 'RS': args.RS}
    fitFile = args.fitfile if args.fitfile else args.filename.split('.')[0] + ".fit"
    try:
        with open(fitFile, 'r') as fh:
            for line in fh:
                name, val = line.split('=')
                params[name.strip()] = float(val.strip())
    except IOError:
        pass

    if args.plot:
        print(f"Plotting with parameters:\nIS={params['IS']:.2e}, N={params['N']:.3f}, RS={params['RS']:.3f}")
        plot(xdata, ydata, args.npoints, params['IS'], params['N'], params['RS'])
        return

    # --- FITTING ---
    try:
        popt, pcov = curve_fit(
            lambda V, IS, N, RS: diode_model(V, IS, N, RS, return_log=not args.linear),
            xdata, ytarget,
            p0=(params['IS'], params['N'], params['RS']),
            bounds=([1e-18, 0.5, 0.01], [1e-8, 5.0, 100.0]),
            maxfev=args.maxit
        )

        print("Fit succeeded with:")
        printString = f"IS = {popt[0]:.4e}\nN = {popt[1]:.4f}\nRS = {popt[2]:.4f}"
        print(printString)

        if args.save:
            outFile = args.filename.split('.')[0] + ".fit"
            with open(outFile, "w") as fh:
                fh.write(printString)

        plot(xdata, ydata, args.npoints, *popt)

    except RuntimeError as e:
        print("❌ Fit did not converge:")
        print(str(e))
        print("📉 Plotting initial guess instead.")
        plot(xdata, ydata, args.npoints, params['IS'], params['N'], params['RS'])

if __name__ == "__main__":
    main()
