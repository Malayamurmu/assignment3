"""Builds report.pdf from results.csv (run run_experiments.py first)."""
import csv
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted

ss = getSampleStyleSheet()
code = ParagraphStyle("code", parent=ss["Code"], fontSize=8.5, leading=11, backColor=colors.whitesmoke)
P = lambda t, s="Normal": Paragraph(t, ss[s])

def tbl(data, widths=None, fs=8):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("FONTSIZE", (0, 0), (-1, -1), fs),
                           ("ALIGN", (1, 1), (-1, -1), "CENTER")]))
    return t

rows = list(csv.reader(open("results.csv")))
s = [P("AI Accelerator Design - Assignment 3", "Title"),
     P("BCHW &harr; B&times;(CHW) Tensor Flattening and Reconstruction", "Heading2"),
     P("Roll No: CS26M205 &nbsp;&nbsp;"), Spacer(1, 8)]

s += [P("1. Address mapping", "Heading2"),
      P("Tensors are stored row-major (C order). For a fixed batch index b, element (c,h,w) lands in column "
        "j of the flattened row, where j = c&middot;(H&middot;W) + h&middot;W + w. The batch index is unchanged. "
        "The inverse recovers c = j // (H&middot;W), r = j mod (H&middot;W), h = r // W, w = r mod W. "
        "No reshape/view/flatten/ravel is used in the algorithm; numpy only allocates arrays and "
        "reads/writes single elements."),
      P("2. Pseudo code", "Heading2")]
s.append(Preformatted(
"""FLATTEN(I[B,C,H,W]):                    RECONSTRUCT(F[B,C*H*W], C,H,W):
  F <- empty(B, C*H*W)                    R <- empty(B,C,H,W)
  for b in 0..B-1:                        for b in 0..B-1:
    for c in 0..C-1:                        for j in 0..C*H*W-1:
      for h in 0..H-1:                        c <- j // (H*W)
        for w in 0..W-1:                      r <- j mod (H*W)
          j <- c*H*W + h*W + w                h <- r // W
          F[b,j] <- I[b,c,h,w]                w <- r mod W
  return F                                    R[b,c,h,w] <- F[b,j]
                                          return R""", code))

s += [P("3. Manual indexing example (B=1, C=2, H=2, W=3, row length 12)", "Heading2")]
man = [["(b,c,h,w)", "j = c*6 + h*3 + w", "j", "value"]]
v = 1
for c in range(2):
    for h in range(2):
        for w in range(3):
            man.append([f"(0,{c},{h},{w})", f"{c}*6 + {h}*3 + {w}", c*6+h*3+w, v]); v += 1
s += [tbl(man, [80, 110, 40, 50]),
      P("Example: element (0,1,0,2) = 9 goes to j = 1&middot;6 + 0&middot;3 + 2 = 8 (the 9th slot, 0-indexed 8). "
        "Inverse: 8 // 6 = 1 (c), 8 mod 6 = 2, 2 // 3 = 0 (h), 2 mod 3 = 2 (w). "), Spacer(1, 6)]

s += [P("4. Experiments and results", "Heading2"),
      tbl([[r[0], *r[1:7]] for r in rows], [120, 30, 35, 30, 30, 55, 55]),
      Spacer(1, 6),
      P("MNIST/CIFAR rows use the real first training image when torchvision and a network are available; "
        "otherwise same-shaped random tensors are used (the row name says which). Synthetic feature maps "
        "use B=2, H=W=16 and C &isin; {1,3,8,16,32,64,128,256,500}. "
        "As an independent check (reference only, outside the algorithm) every flattened result was also "
        "compared with numpy reshape: all matched."),
      P("5. Discussion", "Heading2"),
      P("Every case gives E<sub>max</sub> = 0 and MAE = 0, as expected: flatten/reconstruct only move values "
        "and never compute on them, and the map j = c&middot;HW + h&middot;W + w is a bijection between "
        "{0..C-1}&times;{0..H-1}&times;{0..W-1} and {0..CHW-1}. The result holds from C=1 to C=500. "
        "Run time grows linearly with the number of elements, O(B&middot;C&middot;H&middot;W), as the "
        "loops visit each element exactly once. In hardware, the same mapping lets a systolic array be "
        "fed with contiguous addresses, and the multiply/add form can be implemented with a counter "
        "chain instead of a divider.")]
SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36).build(s)
