import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import symbols, sympify, lambdify, diff, Matrix, det as sym_det, eye as sym_eye
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("blue")
    CTK = True
except ImportError:
    CTK = False

C = {
    "window_bg":        "#FFF0F5",
    "side_panel":       "#FAD0EC",
    "brand":            "#6B3B53",
    "interactive":      "#D482B7",
    "entry_bg":         "#FFF9FC",
    "entry_text":       "#6E3952",
    "entry_border":     "#D482B7",
    "btn_bg":           "#EBC3DA",
    "btn_hover":        "#DFB1C8",
    "tab_bg":           "#EBC3DA",
    "tab_active":       "#6B3B53",
    "table_header":     "#D482B7",
    "table_row":        "#FFF9FC",
    "table_text":       "#52263C",
    "terminal_bg":      "#EBC3DA",
    "terminal_text":    "#52263C",
}

FONT_BODY  = ("Segoe UI", 11)
FONT_BOLD  = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_MONO  = ("Consolas", 10)


# ─────────────────────────────────────────────
#  EXPRESSION PARSER
# ─────────────────────────────────────────────
def fix_expr(s: str) -> str:
    """Normalise common user input into SymPy-parseable string.
    Handles: 3x, sinx, cosx, tanx, e^x, ln, pi, |x|, etc.
    """
    import re
    s = s.strip()

    # Replace ^ with ** (power)
    s = s.replace("^", "**")

    # Replace common named constants/functions written without parens or space
    # e.g. sinx -> sin(x), cosx -> cos(x), tanx -> tan(x)
    # Must do before implicit multiplication so they are not split
    s = re.sub(r'\bsin\s*x\b', 'sin(x)', s)
    s = re.sub(r'\bcos\s*x\b', 'cos(x)', s)
    s = re.sub(r'\btan\s*x\b', 'tan(x)', s)
    s = re.sub(r'\bcsc\s*x\b', 'csc(x)', s)
    s = re.sub(r'\bsec\s*x\b', 'sec(x)', s)
    s = re.sub(r'\bcot\s*x\b', 'cot(x)', s)
    s = re.sub(r'\bln\s*x\b',  'log(x)', s)
    s = re.sub(r'\bln\s*\(', 'log(', s)

    # Replace e** with E** (SymPy E is Euler's number)
    # But only standalone 'e' not inside a word like 'exp' or 'sec'
    s = re.sub(r'(?<![a-df-zA-Z])e(?![a-zA-Z])', 'E', s)

    # Replace pi with explicit symbol
    s = re.sub(r'(?<![a-zA-Z])pi(?![a-zA-Z])', 'pi', s)

    # |expr| absolute value -> Abs(expr)  — simple single-level only
    s = re.sub(r'\|([^|]+)\|', r'Abs(\1)', s)

    return s

def parse_f(s: str):
    from sympy import E, pi as sym_pi, sin, cos, tan, exp, log, sqrt, Abs
    x = symbols("x")
    tf = standard_transformations + (implicit_multiplication_application,)
    local = {
        "x": x, "E": E, "e": E, "pi": sym_pi,
        "sin": sin, "cos": cos, "tan": tan,
        "exp": exp, "log": log, "ln": log,
        "sqrt": sqrt, "Abs": Abs, "abs": Abs,
    }
    return parse_expr(fix_expr(s), transformations=tf, local_dict=local)

def make_callable(s: str):
    x = symbols("x")
    expr = parse_f(s)
    return lambdify(x, expr, modules=["numpy"]), expr


# ═══════════════════════════════════════════════════════════
#  NUMERICAL METHODS
# ═══════════════════════════════════════════════════════════

def bisection(f_str, a, b, tol=1e-5, max_iter=100):
    f, _ = make_callable(f_str)
    fa, fb = float(f(a)), float(f(b))
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
    rows, errors, steps = [], [], []
    xr_old = None
    steps.append("Bisection Method")
    steps.append(f"f({a:.4f})={fa:.6f},  f({b:.4f})={fb:.6f}")
    steps.append(f"f(xL)·f(xU) = {fa*fb:.6f} < 0 ✓  (sign change confirmed)\n")
    for i in range(1, max_iter + 1):
        xr  = (a + b) / 2.0
        fxr = float(f(xr))
        fxa = float(f(a))
        ea  = abs((xr - xr_old) / xr) * 100 if xr_old is not None else None
        ea_disp = f"{ea:.6f}%" if ea is not None else "---"
        if fxa * fxr < 0:
            subint = "1st (xU = xR)"; b = xr
        elif fxa * fxr > 0:
            subint = "2nd (xL = xR)"; a = xr
        else:
            subint = "exact root"
        rows.append((i, f"{a:.6f}", f"{b:.6f}", f"{xr:.6f}", f"{fxr:.6f}", ea_disp))
        errors.append(ea if ea is not None else 0)
        steps += [
            f"Iteration {i}:",
            f"  xL = {a:.6f},  xU = {b:.6f}",
            f"  xR = ({a:.6f} + {b:.6f})/2 = {xr:.6f}",
            f"  f(xR) = {fxr:.6f}",
            f"  |εa| = {ea_disp}",
            f"  f(xL)·f(xR) → Subinterval: {subint}",
            "  " + "─"*35,
        ]
        if ea is not None and ea < tol * 100:
            steps.append(f"\n✓ Converged! Root ≈ {xr:.8f}  (|εa|={ea:.6f}% < εs={tol*100:.4f}%)")
            return xr, f, rows, errors, "\n".join(steps)
        xr_old = xr
    return (a + b) / 2, f, rows, errors, "\n".join(steps)


def regula_falsi(f_str, a, b, tol=1e-5, max_iter=100):
    f, _ = make_callable(f_str)
    if float(f(a)) * float(f(b)) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    rows, errors, steps = [], [], []
    xr_old = None
    steps.append("Regula Falsi (False Position) Method")
    steps.append(f"f(xL)·f(xU) = {float(f(a))*float(f(b)):.6f} < 0 ✓\n")
    for i in range(1, max_iter + 1):
        fxl, fxu = float(f(a)), float(f(b))
        denom = fxl - fxu
        if abs(denom) < 1e-15:
            break
        xr  = (b * fxl - a * fxu) / denom
        fxr = float(f(xr))
        ea  = abs((xr - xr_old) / xr) * 100 if xr_old is not None else None
        ea_disp = f"{ea:.6f}%" if ea is not None else "---"
        if fxl * fxr < 0:
            sign_txt = "< 0 → xU = xR"; b = xr
        elif fxl * fxr > 0:
            sign_txt = "> 0 → xL = xR"; a = xr
        else:
            sign_txt = "= 0 → exact root"
        rows.append((i, f"{a:.6f}", f"{b:.6f}", f"{xr:.6f}", ea_disp, f"{fxl:.6f}", f"{fxu:.6f}", f"{fxr:.6f}"))
        errors.append(ea if ea is not None else 0)
        steps += [
            f"Iteration {i}:",
            f"  xL = {a:.6f},  xU = {b:.6f}",
            f"  xR = [xU·f(xL) - xL·f(xU)] / [f(xL) - f(xU)]",
            f"     = [{b:.4f}×{fxl:.6f} - {a:.4f}×{fxu:.6f}] / [{fxl:.6f} - {fxu:.6f}]",
            f"     = {xr:.6f}",
            f"  f(xL)={fxl:.6f}  f(xU)={fxu:.6f}  f(xR)={fxr:.6f}",
            f"  f(xL)·f(xR) {sign_txt}",
            f"  |εa| = {ea_disp}",
            "  " + "─"*35,
        ]
        if ea is not None and ea < tol * 100:
            steps.append(f"\n✓ Converged! Root ≈ {xr:.8f}  (|εa|={ea:.6f}% < εs={tol*100:.4f}%)")
            return xr, f, rows, errors, "\n".join(steps)
        xr_old = xr
    return xr, f, rows, errors, "\n".join(steps)


def newton_raphson(f_str, x0, tol=1e-5, max_iter=100):
    xs = symbols("x")
    expr = parse_f(f_str)
    d_expr = diff(expr, xs)
    f  = lambdify(xs, expr,   modules=["numpy"])
    df = lambdify(xs, d_expr, modules=["numpy"])
    rows, errors, steps = [], [], []
    steps.append("Newton-Raphson Method")
    steps.append("f(x) and f'(x) computed symbolically.")
    steps.append(f"Initial guess: x0 = {x0:.6f}\n")
    xi = x0
    for i in range(1, max_iter + 1):
        fxi  = float(f(xi))
        dfxi = float(df(xi))
        if abs(dfxi) < 1e-12:
            raise ValueError(f"Zero derivative at x={xi:.6f} — method fails.")
        x_new = xi - fxi / dfxi
        ea = abs((x_new - xi) / x_new) * 100 if abs(x_new) > 1e-15 else abs(x_new - xi) * 100
        rows.append((i, f"{x_new:.6f}", f"{float(f(x_new)):.6f}", f"{ea:.6f}%"))
        errors.append(ea)
        steps += [
            f"Iteration {i}:",
            f"  xi = {xi:.6f}",
            f"  f(xi)  = {fxi:.6f}",
            f"  f'(xi) = {dfxi:.6f}",
            f"  x(i+1) = {xi:.6f} - {fxi:.6f}/{dfxi:.6f}",
            f"         = {x_new:.8f}",
            f"  |εa| = {ea:.6f}%",
            "  " + "─"*35,
        ]
        if ea < tol * 100:
            steps.append(f"\n✓ Converged! Root ≈ {x_new:.8f}  (|εa|={ea:.6f}%)")
            return x_new, f, rows, errors, "\n".join(steps)
        xi = x_new
    return xi, f, rows, errors, "\n".join(steps)


def secant(f_str, x0, x1, tol=1e-5, max_iter=100):
    f, _ = make_callable(f_str)
    rows, errors, steps = [], [], []
    steps.append("Secant Method")
    steps.append(f"Initial guesses: x0 = {x0:.6f},  x1 = {x1:.6f}\n")
    for i in range(1, max_iter + 1):
        fx0, fx1 = float(f(x0)), float(f(x1))
        denom = fx1 - fx0
        if abs(denom) < 1e-12:
            raise ValueError(f"Division by zero in Secant at iteration {i}.")
        x2  = x1 - fx1 * (x1 - x0) / denom
        fx2 = float(f(x2))
        ea  = abs((x2 - x1) / x2) * 100 if abs(x2) > 1e-15 else abs(x2 - x1) * 100
        rows.append((i, f"{x0:.6f}", f"{x1:.6f}", f"{x2:.6f}", f"{ea:.6f}%", f"{fx0:.6f}", f"{fx1:.6f}", f"{fx2:.6f}"))
        errors.append(ea)
        steps += [
            f"Iteration {i}:",
            f"  x(i-1) = {x0:.6f},  x(i) = {x1:.6f}",
            f"  f(x(i-1)) = {fx0:.6f},  f(x(i)) = {fx1:.6f}",
            f"  x(i+1) = x(i) - f(x(i))·[x(i)-x(i-1)] / [f(x(i))-f(x(i-1))]",
            f"         = {x1:.6f} - {fx1:.6f}×({x1:.6f}-{x0:.6f})/({fx1:.6f}-{fx0:.6f})",
            f"         = {x2:.8f}",
            f"  f(x(i+1)) = {fx2:.6f}",
            f"  |εa| = {ea:.6f}%",
            "  " + "─"*35,
        ]
        if ea < tol * 100:
            steps.append(f"\n✓ Converged! Root ≈ {x2:.8f}  (|εa|={ea:.6f}%)")
            return x2, f, rows, errors, "\n".join(steps)
        x0, x1 = x1, x2
    return x1, f, rows, errors, "\n".join(steps)


def fixed_point(f_str, x0, tol=1e-5, max_iter=100):
    """g(x) iteration — user enters g(x) in the f(x) field."""
    f, _ = make_callable(f_str)
    rows, errors, steps = [], [], []
    steps.append("Simple Fixed-Point Iteration")
    steps.append("g(x) as entered  (rearranged so x = g(x))")
    steps.append(f"Initial guess: x0 = {x0:.6f}\n")
    xi = x0
    for i in range(1, max_iter + 1):
        x_new = float(f(xi))
        ea = abs((x_new - xi) / x_new) * 100 if abs(x_new) > 1e-15 else abs(x_new - xi) * 100
        rows.append((i, f"{xi:.6f}", f"{x_new:.6f}", f"{ea:.6f}%"))
        errors.append(ea)
        steps += [
            f"Iteration {i}:",
            f"  x(i) = {xi:.6f}",
            f"  x(i+1) = g(x(i)) = {x_new:.8f}",
            f"  |εa| = {ea:.6f}%",
            "  " + "─"*35,
        ]
        if ea < tol * 100:
            steps.append(f"\n✓ Converged! Root ≈ {x_new:.8f}  (|εa|={ea:.6f}%)")
            return x_new, f, rows, errors, "\n".join(steps)
        xi = x_new
    return xi, f, rows, errors, "\n".join(steps)


def incremental_search(f_str, a, b, tol=1e-5, max_iter=200):
    f, _ = make_callable(f_str)
    h = (b - a) / 20.0
    xl = a
    rows, errors, steps = [], [], []
    steps.append("Incremental Search Method")
    steps.append(f"Interval [{a:.4f}, {b:.4f}],  initial step Δx = {h:.6f}\n")
    refine, max_refine = 0, 5
    for i in range(1, max_iter + 1):
        xu = xl + h
        if xu > b:
            break
        fxl, fxu = float(f(xl)), float(f(xu))
        rows.append((i, f"{xl:.6f}", f"{h:.8f}", f"{xu:.6f}", f"{fxl:.6f}", f"{fxu:.6f}"))
        errors.append(abs(fxu - fxl))
        sign_info = f"{fxl*fxu:.6f}"
        if fxl * fxu < 0:
            steps += [
                f"Iteration {i}:",
                f"  xL={xl:.6f}  Δx={h:.8f}  xU={xu:.6f}",
                f"  f(xL)={fxl:.6f}  f(xU)={fxu:.6f}",
                f"  f(xL)·f(xU)={sign_info} < 0  → Sign change detected!",
            ]
            refine += 1
            if refine <= max_refine:
                h /= 10.0
                steps.append(f"  → Refining: new Δx = {h:.8f}")
                steps.append("  " + "─"*35)
                continue
            else:
                root = (xl + xu) / 2.0
                steps.append(f"  → Root ≈ {root:.8f}")
                steps.append("  " + "─"*35)
                steps.append(f"\n✓ Root found ≈ {root:.8f}")
                return root, f, rows, errors, "\n".join(steps)
        else:
            steps += [
                f"Iteration {i}:",
                f"  xL={xl:.6f}  Δx={h:.8f}  xU={xu:.6f}",
                f"  f(xL)·f(xU)={sign_info} ≥ 0  → No sign change, advance",
                "  " + "─"*35,
            ]
        xl = xu
    return xl, f, rows, errors, "\n".join(steps)


# ─────────────────────────────────────────────
#  FORMULA DESCRIPTIONS  (MATLAB parity)
# ─────────────────────────────────────────────
FORMULAS = {
    "Bisection": """\
BISECTION METHOD  (Bracketing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prerequisite: f(xL)·f(xU) < 0

Root estimate:
  xR = (xL + xU) / 2

Approx. Relative Error:
  |εa| = |(xR_new - xR_old) / xR_new| × 100%

Update rule:
  f(xL)·f(xR) < 0 → xU = xR   (1st subinterval)
  f(xL)·f(xR) > 0 → xL = xR   (2nd subinterval)
  f(xL)·f(xR) = 0 → root is xR, stop

Stop when: |εa| ≤ εs = 0.5×10^(2-n) %""",

    "Regula Falsi": """\
REGULA FALSI / FALSE POSITION  (Bracketing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prerequisite: f(xL)·f(xU) < 0

Root estimate (similar triangles):
  xR = [xU·f(xL) - xL·f(xU)] / [f(xL) - f(xU)]

Approx. Relative Error:
  |εa| = |(xR_new - xR_old) / xR_new| × 100%

Update rule:
  f(xL)·f(xR) < 0 → xU = xR
  f(xL)·f(xR) > 0 → xL = xR
  f(xL)·f(xR) = 0 → exact root

Stop when: |εa| < εs""",

    "Newton-Raphson": """\
NEWTON-RAPHSON METHOD  (Open)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requires: single initial guess x0, f'(x) ≠ 0

Iterative formula:
  x(i+1) = x(i) - f(x(i)) / f'(x(i))

From tangent line slope:
  f'(xi) = [f(xi) - 0] / [xi - x(i+1)]
  → x(i+1) = xi - f(xi)/f'(xi)

Approx. Relative Error:
  |εa| = |(x_new - x_old) / x_new| × 100%

Advantages: Quadratic convergence (fast)
Drawbacks : f'(xi)=0 → division by zero;
            diverges near inflection points""",

    "Secant": """\
SECANT METHOD  (Open)
━━━━━━━━━━━━━━━━━━━━━
Requires: two initial guesses x0 and x1
(does NOT require computing f'(x))

Iterative formula:
  x(i+1) = x(i) - f(x(i))·[x(i) - x(i-1)]
              / [f(x(i)) - f(x(i-1))]

Approx. Relative Error:
  |εa| = |(x_new - x_old) / x_new| × 100%

Stop when: |εa| < εs""",

    "Fixed-Point": """\
SIMPLE FIXED-POINT ITERATION  (Open)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rearrange f(x)=0 into x = g(x)

Iterative formula:
  x(i+1) = g(x(i))

Approx. Relative Error:
  |εa| = |(x(i+1) - x(i)) / x(i+1)| × 100%

Convergence condition : |g'(x)| < 1
Convergence type      : Linear (slower than N-R)

Note: Enter g(x) in the f(x) field.""",

    "Incremental": """\
INCREMENTAL SEARCH  (Bracketing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Steps through [a,b] with step Δx

Sign-change condition:
  f(xL)·f(xU) < 0  →  root in [xL, xU]

Algorithm:
  1. Select xL and Δx;  xU = xL + Δx
  2. If f(xL)·f(xU) < 0 → root is bracketed
  3. Revert to last xL, reduce Δx (refine)
  4. Repeat until |εa| ≤ εs

Root ≈ (xL + xU) / 2  after sufficient refinement""",
}


# ═══════════════════════════════════════════════════════════
#  MATRIX HELPERS
# ═══════════════════════════════════════════════════════════
def fmt_mat(M, prefix="  "):
    lines = []
    for row in M:
        cells = "  ".join(f"{v:10.4f}" for v in row)
        lines.append(f"{prefix}[ {cells} ]")
    return "\n".join(lines)

def adj_matrix(A):
    n = len(A)
    A_sym = Matrix(A)
    cofactors = Matrix(n, n, lambda i, j:
        ((-1) ** (i + j)) * A_sym.minor(i, j))
    return np.array(cofactors.T.tolist(), dtype=float)

def rref_steps(A):
    M = [row[:] for row in A]
    r, c = len(M), len(M[0])
    lines = []
    piv_row = 0
    for col in range(c):
        if piv_row >= r:
            break
        max_val = max(range(piv_row, r), key=lambda i: abs(M[i][col]))
        if abs(M[max_val][col]) < 1e-10:
            lines.append(f"  Column {col+1}: no pivot (skip)")
            continue
        if max_val != piv_row:
            M[piv_row], M[max_val] = M[max_val], M[piv_row]
            lines.append(f"  R{piv_row+1} ↔ R{max_val+1}")
            lines.append(fmt_mat(M, "    "))
        sc = M[piv_row][col]
        if abs(sc) > 1e-10:
            M[piv_row] = [v / sc for v in M[piv_row]]
            lines.append(f"  R{piv_row+1} ÷ {sc:.4f}")
            lines.append(fmt_mat(M, "    "))
        for row in range(r):
            if row != piv_row and abs(M[row][col]) > 1e-10:
                fac = M[row][col]
                M[row] = [M[row][k] - fac * M[piv_row][k] for k in range(c)]
                lines.append(f"  R{row+1} = R{row+1} - ({fac:.4f})×R{piv_row+1}")
                lines.append(fmt_mat(M, "    "))
        piv_row += 1
    return "\n".join(lines), M


def compute_matrix_op(op, A_raw, B_raw, power=2):
    """Returns (result_text, steps_text)."""
    try:
        A = np.array(A_raw, dtype=float)
    except Exception:
        raise ValueError("Matrix A is invalid.")

    rA, cA = A.shape
    result_lines, steps = [], []

    if op == "Addition":
        B = np.array(B_raw, dtype=float)
        if A.shape != B.shape:
            raise ValueError("Matrices must be the same size for Addition.")
        R = A + B
        steps.append("MATRIX ADDITION: R = A + B")
        steps.append("Add corresponding elements:\n")
        for i in range(rA):
            for j in range(cA):
                steps.append(f"  R({i+1},{j+1}) = {A[i,j]:.4f} + {B[i,j]:.4f} = {R[i,j]:.4f}")
        result_lines = [fmt_mat(R)]

    elif op == "Multiplication":
        B = np.array(B_raw, dtype=float)
        if cA != B.shape[0]:
            raise ValueError("Columns of A must equal rows of B.")
        R = A @ B
        rB, cB = B.shape
        steps.append(f"MATRIX MULTIPLICATION: R = A × B")
        steps.append(f"A is {rA}×{cA},  B is {rB}×{cB}  →  R is {rA}×{cB}\n")
        for i in range(rA):
            for j in range(cB):
                terms = " + ".join(f"{A[i,k]:.4f}×{B[k,j]:.4f}" for k in range(cA))
                steps.append(f"  R({i+1},{j+1}) = {terms} = {R[i,j]:.4f}")
        result_lines = [fmt_mat(R)]

    elif op == "Transpose":
        R = A.T
        steps.append("TRANSPOSE: R = Aᵀ")
        steps.append(f"Rule: R(i,j) = A(j,i)")
        steps.append(f"A is {rA}×{cA}  →  Aᵀ is {cA}×{rA}\n")
        for i in range(rA):
            for j in range(cA):
                steps.append(f"  R({j+1},{i+1}) = A({i+1},{j+1}) = {A[i,j]:.4f}")
        result_lines = [fmt_mat(R)]

    elif op == "Inverse":
        if rA != cA:
            raise ValueError("Matrix must be square for Inverse.")
        d = float(np.linalg.det(A))
        if abs(d) < 1e-12:
            raise ValueError("Matrix is singular (det ≈ 0). Inverse does not exist.")
        R = np.linalg.inv(A)
        aug = np.hstack([A, np.eye(rA)])
        steps.append("MATRIX INVERSE: R = A⁻¹\n")
        steps.append(f"Step 1: det(A) = {d:.6f} ≠ 0 ✓\n")
        steps.append("Step 2: Form augmented matrix [A | I] and row-reduce:")
        steps.append(fmt_mat(aug.tolist(), "  ") + "\n")
        step_str, _ = rref_steps(aug.tolist())
        steps.append(step_str)
        steps.append("\nResult A⁻¹ =")
        steps.append(fmt_mat(R))
        result_lines = [fmt_mat(R)]

    elif op == "Determinant":
        if rA != cA:
            raise ValueError("Matrix must be square for Determinant.")
        d = float(np.linalg.det(A))
        steps.append("DETERMINANT: det(A)\n")
        if rA == 2:
            steps.append("For 2×2: det = a11·a22 - a12·a21")
            steps.append(f"  = {A[0,0]:.4f}×{A[1,1]:.4f} - {A[0,1]:.4f}×{A[1,0]:.4f}")
            steps.append(f"  = {A[0,0]*A[1,1]:.6f} - {A[0,1]*A[1,0]:.6f}")
            steps.append(f"  = {d:.6f}")
        elif rA == 3:
            steps.append("For 3×3 (Sarrus / cofactor expansion):")
            steps.append(f"  det = {A[0,0]:.4f}({A[1,1]:.4f}·{A[2,2]:.4f} - {A[1,2]:.4f}·{A[2,1]:.4f})")
            steps.append(f"       -{A[0,1]:.4f}({A[1,0]:.4f}·{A[2,2]:.4f} - {A[1,2]:.4f}·{A[2,0]:.4f})")
            steps.append(f"       +{A[0,2]:.4f}({A[1,0]:.4f}·{A[2,1]:.4f} - {A[1,1]:.4f}·{A[2,0]:.4f})")
            steps.append(f"\n  det(A) = {d:.6f}")
        else:
            steps.append("  (LU decomposition used for large matrix)")
            steps.append(f"  det(A) = {d:.6f}")
        result_lines = [f"det(A) = {d:.8f}"]

    elif op == "Power":
        if rA != cA:
            raise ValueError("Matrix must be square for Power.")
        n = int(power)
        R = np.linalg.matrix_power(A, n)
        steps.append(f"MATRIX POWER: R = A^{n}\n")
        steps.append(f"Repeated multiplication: A×A×...×A  ({n} times)\n")
        steps.append("Result:")
        steps.append(fmt_mat(R))
        result_lines = [fmt_mat(R)]

    elif op == "Adjoint":
        if rA != cA:
            raise ValueError("Matrix must be square for Adjoint.")
        R = adj_matrix(A.tolist())
        steps.append("ADJOINT (Adjugate): adj(A) = Cᵀ\n")
        steps.append("Step 1: Cofactors  C(i,j) = (-1)^(i+j) × det(M_ij)\n")
        for i in range(rA):
            for j in range(cA):
                sub = np.delete(np.delete(A, i, axis=0), j, axis=1)
                mv = float(np.linalg.det(sub))
                cv = ((-1) ** (i + j)) * mv
                steps.append(f"  C({i+1},{j+1}) = (-1)^{i+j+2} × {mv:.4f} = {cv:.4f}")
        steps.append("\nStep 2: Transpose cofactor matrix → adj(A) = Cᵀ\n")
        steps.append("adj(A) =")
        steps.append(fmt_mat(R))
        result_lines = [fmt_mat(R)]

    elif op == "Rank":
        steps.append("MATRIX RANK via RREF\n")
        steps.append("Step 1: Original Matrix A =")
        steps.append(fmt_mat(A) + "\n")
        steps.append("Step 2: Apply Row Reduction (RREF):")
        step_str, M_rref = rref_steps(A.tolist())
        steps.append(step_str)
        steps.append("\nStep 3: RREF(A) =")
        steps.append(fmt_mat(M_rref) + "\n")
        non_zero = sum(1 for row in M_rref if any(abs(v) > 1e-10 for v in row))
        R = int(np.linalg.matrix_rank(A))
        nullity = cA - R
        steps.append(f"Step 4: Non-zero rows = {non_zero}")
        steps.append(f"\n∴ Rank(A) = {R}")
        steps.append(f"   Nullity = {cA} - {R} = {nullity}")
        result_lines = [f"Rank(A) = {R}", f"Nullity = {nullity}"]

    elif op == "Eigenvalues":
        if rA != cA:
            raise ValueError("Matrix must be square for Eigenvalues.")
        evals, evecs = np.linalg.eig(A)
        steps.append("EIGENVALUES & EIGENVECTORS\n")
        steps.append("Definition: Av = λv  →  det(A - λI) = 0\n")
        steps.append("Step 1: Solve characteristic polynomial det(A - λI) = 0\n")
        result_lines = []
        for k, (lam, vec) in enumerate(zip(evals, evecs.T)):
            lam_str = f"{lam.real:.6f}" + (f" + {lam.imag:.4f}i" if abs(lam.imag) > 1e-10 else "")
            vec_str = ", ".join(f"{v.real:.4f}" for v in vec)
            steps.append(f"  λ{k+1} = {lam_str}")
            steps.append(f"  Eigenvector v{k+1} = [{vec_str}]ᵀ\n")
            result_lines.append(f"λ{k+1} = {lam_str}")
        result_lines = result_lines  # already set

    elif op == "Solve AX=B":
        B = np.array(B_raw, dtype=float)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        if A.shape[0] != B.shape[0]:
            raise ValueError("Rows of A must equal rows of B.")
        d = float(np.linalg.det(A))
        if abs(d) < 1e-12:
            raise ValueError("Matrix A is singular — no unique solution.")
        R = np.linalg.solve(A, B)
        aug = np.hstack([A, B])
        steps.append("SOLVE SYSTEM AX = B via RREF\n")
        steps.append(f"Step 1: det(A) = {d:.6f} ≠ 0 ✓\n")
        steps.append("Step 2: Augmented matrix [A | B]:")
        steps.append(fmt_mat(aug.tolist(), "  ") + "\n")
        steps.append("Step 3: RREF [A | B]:")
        step_str, aug_rref = rref_steps(aug.tolist())
        steps.append(step_str)
        steps.append("\nStep 4: Solution:")
        result_lines = []
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                steps.append(f"  x({i+1},{j+1}) = {R[i,j]:.6f}")
                result_lines.append(f"x({i+1},{j+1}) = {R[i,j]:.6f}")
    else:
        raise ValueError(f"Unknown operation: {op}")

    return "\n".join(result_lines), "\n".join(steps)


# ═══════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════
class PinkRibbonMathSuite(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pink Ribbon Math Suite")
        self.geometry("1280x860")
        self.minsize(1100, 740)
        self.configure(bg=C["window_bg"])
        self._build_styles()
        self._build_notebook()
        self._build_root_tab()
        self._build_matrix_tab()
        self._update_method_state()

    # ── TTK STYLES ──────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        # Notebook
        s.configure("Pink.TNotebook",
                    background=C["window_bg"],
                    tabmargins=[4, 4, 0, 0])
        s.configure("Pink.TNotebook.Tab",
                    background=C["btn_bg"],
                    foreground=C["brand"],
                    font=FONT_BOLD,
                    padding=[14, 6],
                    focuscolor=C["window_bg"])
        s.map("Pink.TNotebook.Tab",
              background=[("selected", C["tab_active"]), ("active", C["interactive"])],
              foreground=[("selected", "#ffffff")])
        # Treeview
        s.configure("Pink.Treeview",
                    background=C["table_row"],
                    fieldbackground=C["table_row"],
                    foreground=C["table_text"],
                    rowheight=26,
                    font=FONT_BODY)
        s.configure("Pink.Treeview.Heading",
                    background=C["table_header"],
                    foreground="#ffffff",
                    font=FONT_BOLD)
        s.map("Pink.Treeview.Heading",
              background=[("active", C["brand"])])
        s.map("Pink.Treeview",
              background=[("selected", C["interactive"])],
              foreground=[("selected", "#ffffff")])
        # Scrollbars
        s.configure("Pink.Vertical.TScrollbar",
                    background=C["btn_bg"], troughcolor=C["window_bg"],
                    arrowcolor=C["brand"])
        # Frame
        s.configure("Card.TFrame", background=C["side_panel"], relief="flat")
        s.configure("Main.TFrame", background=C["window_bg"], relief="flat")

    # ── NOTEBOOK ────────────────────────────────────────────────────────────
    def _build_notebook(self):
        self.nb = ttk.Notebook(self, style="Pink.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.root_tab   = ttk.Frame(self.nb, style="Main.TFrame")
        self.matrix_tab = ttk.Frame(self.nb, style="Main.TFrame")
        self.nb.add(self.root_tab,   text="📐  Root Finding")
        self.nb.add(self.matrix_tab, text="🔢  Linear Algebra")

    # ════════════════════════════════════════════════════════
    #  TAB 1 — ROOT FINDING
    # ════════════════════════════════════════════════════════
    def _build_root_tab(self):
        tab = self.root_tab

        # Use a PanedWindow so user can resize top (graph) vs bottom (table+panels)
        paned = tk.PanedWindow(tab, orient="vertical", bg=C["window_bg"],
                               sashwidth=6, sashrelief="flat",
                               sashpad=2)
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        # ── TOP PANE: sidebar + graph ──────────────────────
        top = ttk.Frame(paned, style="Main.TFrame")
        paned.add(top, minsize=260, stretch="always")

        # LEFT SIDEBAR — scrollable so nothing is clipped
        sidebar_outer = tk.Frame(top, bg=C["side_panel"], bd=0, width=272)
        sidebar_outer.pack(side="left", fill="y", padx=(0, 6))
        sidebar_outer.pack_propagate(False)

        sb_canvas = tk.Canvas(sidebar_outer, bg=C["side_panel"],
                              highlightthickness=0, bd=0)
        sb_scrollbar = ttk.Scrollbar(sidebar_outer, orient="vertical",
                                     command=sb_canvas.yview,
                                     style="Pink.Vertical.TScrollbar")
        sidebar = tk.Frame(sb_canvas, bg=C["side_panel"])

        sidebar_win = sb_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sb_canvas.configure(yscrollcommand=sb_scrollbar.set)

        def _on_sidebar_configure(event):
            sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))
            sb_canvas.itemconfig(sidebar_win, width=sb_canvas.winfo_width())

        sidebar.bind("<Configure>", _on_sidebar_configure)
        sb_canvas.pack(side="left", fill="both", expand=True)
        # Only show scrollbar when needed — bind mousewheel
        def _sidebar_scroll(event):
            sb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        sb_canvas.bind_all("<MouseWheel>", _sidebar_scroll)

        def lbl(text, **kw):
            w = tk.Label(sidebar, text=text, bg=C["side_panel"],
                         fg=C["brand"], font=FONT_BOLD, **kw)
            w.pack(anchor="w", padx=12, pady=(4, 0))
            return w

        def entry(default="", width=26):
            e = tk.Entry(sidebar, width=width,
                         bg=C["entry_bg"], fg=C["entry_text"],
                         insertbackground=C["brand"],
                         relief="flat", bd=2,
                         font=FONT_BODY,
                         highlightbackground=C["entry_border"],
                         highlightcolor=C["interactive"],
                         highlightthickness=1)
            e.pack(fill="x", padx=12, pady=2)
            e.insert(0, default)
            return e

        lbl("Method:")
        self.method_var = tk.StringVar(value="Bisection")
        method_menu = tk.OptionMenu(
            sidebar, self.method_var,
            "Bisection", "Regula Falsi", "Newton-Raphson",
            "Secant", "Fixed-Point", "Incremental",
            command=lambda _: self._update_method_state()
        )
        method_menu.config(
            bg=C["brand"], fg="#ffffff", activebackground=C["interactive"],
            activeforeground="#ffffff", relief="flat", bd=0,
            font=FONT_BOLD, highlightthickness=0, width=24)
        method_menu["menu"].config(
            bg=C["btn_bg"], fg=C["brand"], font=FONT_BODY,
            activebackground=C["interactive"], activeforeground="#ffffff")
        method_menu.pack(fill="x", padx=12, pady=2)

        lbl("f(x)  or  g(x):")
        self.eq_entry = entry("x**3 - x - 1")

        # bracketing inputs — keep references to both label AND entry as a pair
        self.lbl_a    = lbl("a  (xL):")
        self.a_entry  = entry("1")
        self.lbl_b    = lbl("b  (xU):")
        self.b_entry  = entry("2")
        # open method inputs
        self.lbl_x0   = lbl("x0  (initial guess):")
        self.x0_entry = entry("1.5")
        self.lbl_x1   = lbl("x1  (2nd guess — Secant):")
        self.x1_entry = entry("2")

        lbl("Tolerance:")
        self.tol_entry  = entry("1e-5")
        lbl("Max Iterations:")
        self.iter_entry = entry("100")

        # Compare checkbox
        self.compare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sidebar, text=" Compare bracketing methods",
                       variable=self.compare_var,
                       bg=C["side_panel"], fg=C["brand"],
                       activebackground=C["side_panel"],
                       selectcolor=C["entry_bg"],
                       font=FONT_BODY).pack(anchor="w", padx=12, pady=(4, 2))

        tk.Button(sidebar, text="▶  Compute & Plot",
                  bg=C["brand"], fg="#ffffff",
                  activebackground=C["interactive"],
                  activeforeground="#ffffff",
                  font=FONT_TITLE, relief="flat", bd=0,
                  cursor="hand2", pady=8,
                  command=self._compute_root).pack(
            fill="x", padx=12, pady=6)

        self.status_lbl = tk.Label(sidebar, text="Convergence Status:\nIdle",
                                   bg=C["side_panel"], fg=C["brand"],
                                   font=FONT_BODY, justify="left", wraplength=230)
        self.status_lbl.pack(anchor="w", padx=12, pady=(4, 8))

        # RIGHT AREA: graph
        right = ttk.Frame(top, style="Main.TFrame")
        right.pack(side="right", fill="both", expand=True)

        self.graph_frame = tk.Frame(right, bg=C["window_bg"], bd=0,
                                    highlightbackground=C["entry_border"],
                                    highlightthickness=1)
        self.graph_frame.pack(fill="both", expand=True)

        # ── BOTTOM PANE: table + formula + steps ──────────
        bottom_pane = ttk.Frame(paned, style="Main.TFrame")
        paned.add(bottom_pane, minsize=280, stretch="always")

        # Table
        table_card = tk.Frame(bottom_pane, bg=C["window_bg"])
        table_card.pack(fill="x", padx=2, pady=(4, 0))
        tk.Label(table_card,
                 text="Numerical Convergence Iteration Log",
                 bg=C["window_bg"], fg=C["brand"], font=FONT_TITLE,
                 anchor="w").pack(anchor="w", padx=8, pady=(4, 2))

        tree_frame = tk.Frame(table_card, bg=C["window_bg"])
        tree_frame.pack(fill="x", padx=8, pady=(0, 4))
        self.tree_cols = ("iter", "c1", "c2", "c3", "c4", "c5", "c6", "c7")
        self.tree = ttk.Treeview(tree_frame, columns=self.tree_cols,
                                  show="headings", height=6,
                                  style="Pink.Treeview")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self.tree.yview,
                             style="Pink.Vertical.TScrollbar")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                             command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="x", expand=True)
        self._set_tree_headers_bisection()

        # Formula + Steps panels
        panel_row = ttk.Frame(bottom_pane, style="Main.TFrame")
        panel_row.pack(fill="both", expand=True, padx=2, pady=(0, 6))

        def text_panel(parent, title):
            f = tk.Frame(parent, bg=C["terminal_bg"], bd=0)
            f.pack(side="left", fill="both", expand=True, padx=(0, 4))
            tk.Label(f, text=title, bg=C["terminal_bg"],
                     fg=C["brand"], font=FONT_BOLD, anchor="w").pack(
                anchor="w", padx=8, pady=(6, 2))
            t = tk.Text(f, bg=C["terminal_bg"], fg=C["terminal_text"],
                        font=FONT_MONO, relief="flat", bd=4,
                        wrap="word", state="disabled")
            sb = ttk.Scrollbar(f, command=t.yview,
                                style="Pink.Vertical.TScrollbar")
            t.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            t.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            return t

        self.formula_text = text_panel(panel_row, "Method Formula & Description")
        self.steps_text   = text_panel(panel_row, "Step-by-Step Solution Log")

        self._update_formula()

    def _set_text(self, widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _set_tree_headers_bisection(self):
        for col in self.tree_cols:
            self.tree.heading(col, text="")
            self.tree.column(col, width=10)
        headers = ["Iter", "xL", "xU", "xR", "f(xR)", "|εa|%", "", ""]
        widths  = [55, 100, 100, 100, 100, 90, 0, 0]
        for col, h, w in zip(self.tree_cols, headers, widths):
            self.tree.heading(col, text=h)
            self.tree.column(col, width=w, anchor="center", minwidth=w)

    def _set_tree_headers(self, names):
        widths = [55] + [105] * (len(names) - 1)
        # pad to 8 cols
        while len(names) < 8: names.append("")
        while len(widths) < 8: widths.append(0)
        for col, h, w in zip(self.tree_cols, names, widths):
            self.tree.heading(col, text=h)
            self.tree.column(col, width=w, anchor="center", minwidth=0)

    def _update_method_state(self):
        m = self.method_var.get()
        bracketing  = m in ("Bisection", "Regula Falsi", "Incremental")
        open_method = m in ("Newton-Raphson", "Fixed-Point")
        secant_only = m == "Secant"

        def show_pair(lbl_w, ent_w):
            lbl_w.pack(anchor="w", padx=12, pady=(4, 0))
            ent_w.pack(fill="x", padx=12, pady=2)

        def hide_pair(lbl_w, ent_w):
            lbl_w.pack_forget()
            ent_w.pack_forget()

        if bracketing:
            show_pair(self.lbl_a, self.a_entry)
            show_pair(self.lbl_b, self.b_entry)
        else:
            hide_pair(self.lbl_a, self.a_entry)
            hide_pair(self.lbl_b, self.b_entry)

        if open_method or secant_only:
            show_pair(self.lbl_x0, self.x0_entry)
        else:
            hide_pair(self.lbl_x0, self.x0_entry)

        if secant_only:
            show_pair(self.lbl_x1, self.x1_entry)
        else:
            hide_pair(self.lbl_x1, self.x1_entry)

        self._update_formula()

    def _update_formula(self):
        m = self.method_var.get()
        self._set_text(self.formula_text,
                       FORMULAS.get(m, "Select a method to see its formula."))

    # ── COMPUTE ROOT ────────────────────────────────────────
    def _compute_root(self):
        try:
            method  = self.method_var.get()
            f_str   = self.eq_entry.get().strip()
            tol     = float(self.tol_entry.get())
            max_it  = int(self.iter_entry.get())

            for item in self.tree.get_children():
                self.tree.delete(item)
            for w in self.graph_frame.winfo_children():
                w.destroy()

            compare = self.compare_var.get()
            methods_to_run = (["Bisection", "Regula Falsi", "Incremental"]
                              if compare else [method])

            fig, (ax_f, ax_e) = plt.subplots(
                1, 2, figsize=(9, 3.4),
                facecolor=C["window_bg"])
            ax_f.set_facecolor("#ffffff")
            ax_e.set_facecolor("#ffffff")

            # Plot base function
            if method in ("Newton-Raphson", "Fixed-Point", "Secant"):
                x0 = float(self.x0_entry.get())
                xlo, xhi = x0 - 3, x0 + 3
            else:
                a_v = float(self.a_entry.get())
                b_v = float(self.b_entry.get())
                xlo, xhi = a_v - 0.5, b_v + 0.5
            xs = np.linspace(xlo, xhi, 600)
            f_call, _ = make_callable(f_str)
            try:
                ys = f_call(xs)
                ax_f.plot(xs, ys, color=C["brand"], lw=2, label="f(x)")
            except Exception:
                pass
            ax_f.axhline(0, color="#888", lw=0.8, ls="--")
            ax_f.grid(True, ls=":", alpha=0.5)
            ax_f.set_title("Function & Root(s)", color=C["brand"], fontsize=10)
            ax_f.set_xlabel("x"); ax_f.set_ylabel("f(x)")

            all_steps = []
            last_rows = []
            last_method = methods_to_run[0]
            colors = ["#C71585", "#6B3B53", "#D482B7"]

            for mi, meth in enumerate(methods_to_run):
                clr = colors[mi % len(colors)]
                a_v = float(self.a_entry.get()) if hasattr(self, "a_entry") else 1.0
                b_v = float(self.b_entry.get()) if hasattr(self, "b_entry") else 2.0
                try:
                    x0_v = float(self.x0_entry.get())
                except: x0_v = 1.5
                try:
                    x1_v = float(self.x1_entry.get())
                except: x1_v = 2.0

                if meth == "Bisection":
                    root, fh, rows, errs, stext = bisection(f_str, a_v, b_v, tol, max_it)
                elif meth == "Regula Falsi":
                    root, fh, rows, errs, stext = regula_falsi(f_str, a_v, b_v, tol, max_it)
                elif meth == "Newton-Raphson":
                    root, fh, rows, errs, stext = newton_raphson(f_str, x0_v, tol, max_it)
                elif meth == "Secant":
                    root, fh, rows, errs, stext = secant(f_str, x0_v, x1_v, tol, max_it)
                elif meth == "Fixed-Point":
                    root, fh, rows, errs, stext = fixed_point(f_str, x0_v, tol, max_it)
                else:
                    root, fh, rows, errs, stext = incremental_search(f_str, a_v, b_v, tol, max_it)

                ax_f.scatter([root], [float(fh(root))], color=clr,
                             zorder=6, s=60, label=f"{meth} root")
                ax_e.plot(errs, "-o", color=clr, lw=1.5,
                          ms=4, label=meth)
                all_steps.append(f"{'═'*35}\n  {meth}\n{'═'*35}")
                all_steps.append(stext)
                last_rows = rows
                last_method = meth

            ax_f.legend(fontsize=8)
            ax_e.set_title("Approx. Relative Error / Iteration",
                           color=C["brand"], fontsize=10)
            ax_e.set_xlabel("Iteration")
            ax_e.set_ylabel("|εa| (%)")
            ax_e.legend(fontsize=8)
            ax_e.grid(True, ls=":", alpha=0.5)
            fig.tight_layout(pad=1.2)

            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Populate table
            if last_method in ("Regula Falsi", "Secant"):
                self._set_tree_headers(
                    ["Iter", "xL/x(i-1)", "xU/x(i)", "xR/x(i+1)",
                     "|εa|%", "f(xL)", "f(xU)", "f(xR)"])
            elif last_method == "Incremental":
                self._set_tree_headers(
                    ["Iter", "xL", "Δh", "xU", "f(xL)", "f(xU)", "", ""])
            elif last_method == "Newton-Raphson":
                self._set_tree_headers(
                    ["Iter", "x(i+1)", "f(x)", "|εa|%", "", "", "", ""])
            elif last_method in ("Bisection",):
                self._set_tree_headers(
                    ["Iter", "xL", "xU", "xR", "f(xR)", "|εa|%", "", ""])
            else:
                self._set_tree_headers(
                    ["Iter", "x(i)", "x(i+1)", "|εa|%", "", "", "", ""])

            for row in last_rows:
                padded = list(row) + [""] * (8 - len(row))
                self.tree.insert("", "end", values=padded[:8])

            root_val = last_rows[-1][1] if last_rows else "N/A"
            self.status_lbl.configure(
                text=f"Convergence Status:\n✓ SUCCESS\n\nRoot ≈ {root:.8f}")
            self._set_text(self.steps_text, "\n".join(all_steps))
            self._update_formula()

        except Exception as ex:
            self.status_lbl.configure(
                text=f"Convergence Status:\n✗ ERROR\n\n{str(ex)}")
            messagebox.showerror("Computation Error", str(ex))

    # ════════════════════════════════════════════════════════
    #  TAB 2 — LINEAR ALGEBRA
    # ════════════════════════════════════════════════════════
    def _build_matrix_tab(self):
        tab = self.matrix_tab

        outer = ttk.Frame(tab, style="Main.TFrame")
        outer.pack(fill="both", expand=True, padx=6, pady=6)

        # LEFT: controls ──────────────────────────────────
        ctrl = tk.Frame(outer, bg=C["side_panel"], width=310)
        ctrl.pack(side="left", fill="y", padx=(0, 6))
        ctrl.pack_propagate(False)

        def sect(text):
            tk.Label(ctrl, text=text, bg=C["side_panel"],
                     fg=C["brand"], font=FONT_BOLD).pack(
                anchor="w", padx=12, pady=(10, 2))

        def mat_entry(parent, default=""):
            e = tk.Entry(parent, width=28,
                         bg=C["entry_bg"], fg=C["entry_text"],
                         relief="flat", bd=4, font=FONT_MONO,
                         insertbackground=C["brand"],
                         highlightbackground=C["entry_border"],
                         highlightcolor=C["interactive"],
                         highlightthickness=1)
            e.pack(fill="x", padx=12, pady=3)
            e.insert(0, default)
            return e

        sect("Matrix A  (e.g. 1 2; 3 4)")
        self.mat_a = mat_entry(ctrl, "1 2; 3 4")
        sect("Matrix B  (for Add / Mult / Solve AX=B)")
        self.mat_b = mat_entry(ctrl, "5 6; 7 8")
        sect("Operation:")
        self.mat_op = tk.StringVar(value="Addition")
        op_menu = tk.OptionMenu(
            ctrl, self.mat_op,
            "Addition", "Multiplication", "Transpose",
            "Inverse", "Determinant", "Power",
            "Adjoint", "Rank", "Eigenvalues", "Solve AX=B",
            command=lambda _: self._update_matrix_state())
        op_menu.config(
            bg=C["brand"], fg="#ffffff", relief="flat", bd=0,
            font=FONT_BOLD, width=22, activebackground=C["interactive"],
            activeforeground="#ffffff", highlightthickness=0)
        op_menu["menu"].config(
            bg=C["btn_bg"], fg=C["brand"], font=FONT_BODY,
            activebackground=C["interactive"], activeforeground="#ffffff")
        op_menu.pack(fill="x", padx=12, pady=4)

        sect("Power  n  (for Power op):")
        self.mat_power = tk.Entry(ctrl, width=10,
                                  bg=C["entry_bg"], fg=C["entry_text"],
                                  relief="flat", bd=4, font=FONT_BODY,
                                  highlightbackground=C["entry_border"],
                                  highlightthickness=1)
        self.mat_power.pack(anchor="w", padx=12, pady=3)
        self.mat_power.insert(0, "2")

        tk.Button(ctrl, text="▶  Compute",
                  bg=C["brand"], fg="#ffffff",
                  activebackground=C["interactive"],
                  activeforeground="#ffffff",
                  font=FONT_TITLE, relief="flat", bd=0,
                  cursor="hand2", pady=8,
                  command=self._compute_matrix).pack(
            fill="x", padx=12, pady=14)

        # MIDDLE: result ──────────────────────────────────
        mid = tk.Frame(outer, bg=C["terminal_bg"])
        mid.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Label(mid, text="Result", bg=C["terminal_bg"],
                 fg=C["brand"], font=FONT_TITLE, anchor="w").pack(
            anchor="w", padx=10, pady=(8, 2))
        self.mat_result = tk.Text(mid, bg=C["terminal_bg"],
                                   fg=C["terminal_text"], font=FONT_MONO,
                                   relief="flat", bd=4, state="disabled",
                                   wrap="word")
        sb_r = ttk.Scrollbar(mid, command=self.mat_result.yview,
                              style="Pink.Vertical.TScrollbar")
        self.mat_result.configure(yscrollcommand=sb_r.set)
        sb_r.pack(side="right", fill="y")
        self.mat_result.pack(fill="both", expand=True, padx=8, pady=8)

        # RIGHT: steps ────────────────────────────────────
        rgt = tk.Frame(outer, bg=C["terminal_bg"])
        rgt.pack(side="right", fill="both", expand=True)
        tk.Label(rgt, text="Step-by-Step Solution", bg=C["terminal_bg"],
                 fg=C["brand"], font=FONT_TITLE, anchor="w").pack(
            anchor="w", padx=10, pady=(8, 2))
        self.mat_steps = tk.Text(rgt, bg=C["terminal_bg"],
                                  fg=C["terminal_text"], font=FONT_MONO,
                                  relief="flat", bd=4, state="disabled",
                                  wrap="word")
        sb_s = ttk.Scrollbar(rgt, command=self.mat_steps.yview,
                              style="Pink.Vertical.TScrollbar")
        self.mat_steps.configure(yscrollcommand=sb_s.set)
        sb_s.pack(side="right", fill="y")
        self.mat_steps.pack(fill="both", expand=True, padx=8, pady=8)

    def _update_matrix_state(self):
        pass  # could grey out B for single-matrix ops

    def _parse_matrix(self, text):
        rows = [r.strip() for r in text.strip().split(";") if r.strip()]
        return [[float(v) for v in r.split()] for r in rows]

    def _compute_matrix(self):
        try:
            A = self._parse_matrix(self.mat_a.get())
            try:
                B = self._parse_matrix(self.mat_b.get())
            except Exception:
                B = [[0]]
            op  = self.mat_op.get()
            pwr = int(self.mat_power.get()) if self.mat_power.get().strip() else 2
            result, steps = compute_matrix_op(op, A, B, power=pwr)
            self._set_text(self.mat_result, result)
            self._set_text(self.mat_steps, steps)
        except Exception as ex:
            self._set_text(self.mat_result, f"Error: {ex}")
            self._set_text(self.mat_steps, f"Error: {ex}")


# ── ENTRY POINT ─────────────────────────────────────────────
if __name__ == "__main__":
    app = PinkRibbonMathSuite()
    app.mainloop()