"""Two symbol-led schematics for the CSE-DKAN-FV mechanism and trust logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path("paper_build/latex/figures/revised_principle_set/flow_and_constraints")

INK = "#171C24"
GREY = "#68727D"
LIGHT = "#F7F9FB"
BLUE = "#337BB7"
BLUE_LIGHT = "#DDECF7"
RED = "#C94E5B"
RED_LIGHT = "#F7E1E4"
MAGENTA = "#A94C76"
MAGENTA_LIGHT = "#F0DFE8"
GOLD = "#C78A22"
GOLD_LIGHT = "#F7EACF"
GREEN = "#4C8F5A"
GREEN_LIGHT = "#DFEEDF"
VIOLET = "#7657A6"
VIOLET_LIGHT = "#EAE2F2"


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
})


def canvas(size=(13.0, 5.5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def rounded(ax, x, y, w, h, fc="white", ec=GREY, lw=1.4, r=0.018, z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.006,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def pill(ax, x, y, s, fc, ec, w=.085, h=.030, fs=5.8):
    rounded(ax, x-w/2, y-h/2, w, h, fc, ec, .85, .009, z=7)
    txt(ax, x, y, s, fs, ec, "bold", z=10)


def txt(ax, x, y, s, fs=8, c=INK, weight="normal", ha="center", va="center", z=9):
    return ax.text(x, y, s, fontsize=fs, color=c, fontweight=weight,
                   ha=ha, va=va, zorder=z)


def arr(ax, a, b, c=INK, lw=1.6, ms=12, rad=0, ls="-", style="-|>", z=7):
    p = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=c,
                        linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=z)
    ax.add_patch(p)
    return p


def wave(ax, x, y, w, h, kind="smooth", c=BLUE, lw=2.0, z=8):
    xx = np.linspace(0, 1, 180)
    if kind == "smooth":
        yy = 0.50 + 0.29 * np.tanh((0.52 - xx) * 5.5)
    elif kind == "jump":
        yy = np.where(xx < 0.55, 0.78, 0.22)
    elif kind == "blend":
        yy = 0.50 + 0.29 * np.tanh((0.54 - xx) * 24)
    elif kind == "weno":
        yy = 0.50 + 0.31 * np.tanh((0.54 - xx) * 15) + 0.045 * np.exp(-100*(xx-.55)**2)
    elif kind == "unsafe":
        yy = 0.50 + 0.28 * np.tanh((0.53 - xx) * 18) + 0.16*np.sin(7*np.pi*xx)*np.exp(-8*(xx-.55)**2)
    elif kind == "filtered1":
        yy = 0.50 + 0.28 * np.tanh((0.53 - xx) * 15) + 0.09*np.sin(7*np.pi*xx)*np.exp(-8*(xx-.55)**2)
    elif kind == "filtered2":
        yy = 0.50 + 0.28 * np.tanh((0.53 - xx) * 11) + 0.035*np.sin(7*np.pi*xx)*np.exp(-8*(xx-.55)**2)
    else:
        raise ValueError(kind)
    ax.plot(x + w*xx, y + h*yy, color=c, lw=lw, solid_capstyle="round", zorder=z)


def lock(ax, x, y, c=GREY, scale=1.0):
    ax.add_patch(Rectangle((x-.008*scale, y-.008*scale), .016*scale, .014*scale,
                           facecolor="white", edgecolor=c, lw=.9, zorder=10))
    ax.add_patch(Arc((x, y+.006*scale), .013*scale, .016*scale,
                     theta1=0, theta2=180, color=c, lw=.9, zorder=10))


def network(ax, x, y, w, h, c=MAGENTA):
    layers = [[(x+.15*w, y+.30*h), (x+.15*w, y+.70*h)],
              [(x+.50*w, y+.22*h), (x+.50*w, y+.50*h), (x+.50*w, y+.78*h)],
              [(x+.85*w, y+.50*h)]]
    for left, right in zip(layers[:-1], layers[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=.7, alpha=.55, zorder=4)
    for layer in layers:
        for p in layer:
            ax.add_patch(Circle(p, .008, facecolor="white", edgecolor=c, lw=.9, zorder=8))


def shield(ax, x, y, symbol, fs=6.8):
    r = .036
    pts = np.array([[x-r*.80,y+r*.55],[x,y+r*.88],[x+r*.80,y+r*.55],
                    [x+r*.65,y-r*.20],[x,y-r*.82],[x-r*.65,y-r*.20]])
    ax.add_patch(Polygon(pts, closed=True, facecolor=GREEN_LIGHT,
                         edgecolor=GREEN, lw=1.45, zorder=5))
    txt(ax, x, y, symbol, fs, GREEN, "bold")


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT/f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT/f"{stem}.png", dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT/f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1_train_then_deploy():
    fig, ax = canvas((13.0, 5.65))

    # Offline: learn only a scalar gate from high-fidelity local records.
    rounded(ax, .025, .705, .95, .245, VIOLET_LIGHT, VIOLET, 1.35, .025)
    txt(ax, .045, .925, "a", 11, INK, "bold")
    txt(ax, .075, .925, "OFFLINE", 7.5, VIOLET, "bold", ha="left")
    lock(ax, .943, .925, VIOLET, .9)
    txt(ax, .925, .925, r"FV", 6.8, VIOLET, "bold", ha="right")

    # Compact role labels make the training logic readable without a prose legend.
    for x,lab in [(.122,"local records"),(.305,"basis pair"),(.49,"oracle blend"),
                  (.667,"trainable gate"),(.858,"held-out loss")]:
        txt(ax,x,.907,lab,5.7,VIOLET,"bold")

    rounded(ax, .065, .765, .115, .105, "white", VIOLET, 1.2)
    wave(ax, .082, .785, .080, .064, "weno", RED, 1.8)
    txt(ax, .122, .748, "WENO", 6.8, VIOLET, "bold")

    arr(ax, (.18,.818), (.225,.818), VIOLET, 1.4, 11)
    rounded(ax, .225, .755, .16, .125, GOLD_LIGHT, GOLD, 1.25)
    wave(ax, .245, .79, .052, .060, "smooth", BLUE, 1.55)
    wave(ax, .315, .79, .052, .060, "jump", RED, 1.55)
    txt(ax, .305, .74, r"$q^{MC}$  ↔  $q^{jump}$", 6.8, GOLD, "bold")

    arr(ax, (.385,.818), (.43,.818), VIOLET, 1.4, 11)
    rounded(ax, .43, .755, .12, .125, "white", VIOLET, 1.2)
    ax.plot([.455,.525],[.81,.81], color=BLUE, lw=2)
    ax.plot([.49,.525],[.81,.81], color=RED, lw=2)
    ax.add_patch(Circle((.49,.81), .012, facecolor="white", edgecolor=MAGENTA, lw=1.5, zorder=9))
    txt(ax, .49, .75, r"$\beta_i^{oracle}$", 8.2, VIOLET, "bold")

    arr(ax, (.55,.818), (.595,.818), VIOLET, 1.4, 11)
    rounded(ax, .595, .745, .145, .145, MAGENTA_LIGHT, MAGENTA, 1.4)
    network(ax, .61, .77, .085, .085, MAGENTA)
    txt(ax, .716, .818, r"$g_{\theta}$", 8.2, MAGENTA, "bold")

    arr(ax, (.74,.818), (.80,.818), VIOLET, 1.4, 11)
    rounded(ax, .80, .765, .115, .105, "white", VIOLET, 1.2)
    ax.add_patch(Circle((.832,.818), .019, facecolor="white", edgecolor=VIOLET, lw=1.1, zorder=7))
    ax.add_patch(Circle((.832,.818), .010, facecolor=VIOLET_LIGHT, edgecolor=VIOLET, lw=1.0, zorder=8))
    ax.plot([.832,.855],[.818,.842], color=VIOLET, lw=1.2, zorder=8)
    txt(ax, .885, .818, r"$\mathcal{L}_g$", 9.2, VIOLET, "bold")
    arr(ax, (.86,.765), (.69,.745), MAGENTA, 1.1, 10, rad=.24, ls="--")
    txt(ax,.775,.735,r"$\nabla_{\theta}\mathcal{L}_g$",5.8,MAGENTA,"bold")

    # Online: fixed conservative evolution plus frozen gated subcell reconstruction.
    rounded(ax, .025, .075, .95, .575, "white", "#B8C3CD", 1.15, .025)
    txt(ax, .045, .625, "b", 11, INK, "bold")
    txt(ax, .075, .625, "ONLINE", 7.5, BLUE, "bold", ha="left")

    # Arrow semantics: state is solid; learned checkpoint transfer is dashed.
    ax.plot([.72,.75],[.625,.625],color=BLUE,lw=1.5,zorder=8)
    arr(ax,(.745,.625),(.755,.625),BLUE,1.3,8)
    txt(ax,.79,.625,"state path",5.6,BLUE,"bold")
    ax.plot([.845,.875],[.625,.625],color=VIOLET,lw=1.2,ls="--",zorder=8)
    txt(ax,.91,.625,r"$\theta_g$ transfer",5.6,VIOLET,"bold")

    blocks = [
        (.055,.38,.105,.14,BLUE_LIGHT,BLUE,"cells"),
        (.195,.38,.11,.14,BLUE_LIGHT,BLUE,"HLLC"),
        (.34,.38,.11,.14,BLUE_LIGHT,BLUE,r"$\bar{\mathbf{U}}^{n+1}$"),
        (.485,.38,.10,.14,LIGHT,GREY,r"$\mathbf{z}_i$"),
        (.62,.37,.12,.16,MAGENTA_LIGHT,MAGENTA,""),
        (.785,.37,.095,.16,GOLD_LIGHT,GOLD,r"$q^*$"),
    ]
    for x,y,w,h,fc,ec,label in blocks:
        rounded(ax,x,y,w,h,fc,ec,1.35,.018)
        txt(ax,x+w/2,y+h/2,label,8.4,ec,"bold")
    for a,b,c in [(.16,.195,BLUE),(.305,.34,BLUE),(.45,.485,BLUE),(.585,.62,MAGENTA),(.74,.785,GOLD)]:
        arr(ax,(a,.45),(b,.45),c,1.55,12)

    for k,hh in enumerate([.78,.64,.36,.28]):
        ax.add_patch(Rectangle((.07+.021*k,.405),.016,.075*hh,facecolor=BLUE,edgecolor="none",alpha=.72,zorder=7))
    arr(ax,(.215,.47),(.287,.47),BLUE,1.15,9)
    txt(ax,.251,.493,r"$\widehat F_{i+1/2}$",6.5,BLUE,"bold")
    lock(ax,.216,.496,BLUE,.78)
    lock(ax,.363,.496,BLUE,.78)

    for x,lab,col in [(.107,"state",BLUE),(.25,"shared flux",BLUE),(.395,"cell mean",BLUE),
                      (.535,"features",GREY),(.68,"frozen gate",MAGENTA),(.832,"subcells",GOLD)]:
        txt(ax,x,.545,lab,5.7,col,"bold")

    # Compact feature/sensor glyphs.
    for x,c in [(.508,RED),(.536,GOLD),(.564,BLUE)]:
        ax.add_patch(Circle((x,.435),.011,facecolor="white",edgecolor=c,lw=1.0,zorder=8))
        ax.plot([x,x+.006],[.435,.444],color=c,lw=1.1,zorder=9)
    txt(ax,.508,.405,r"$\Delta p$",4.9,RED,"bold")
    txt(ax,.536,.405,r"$[-\Delta u]_+$",4.7,GOLD,"bold")
    txt(ax,.564,.405,r"$r_q$",4.9,BLUE,"bold")
    network(ax,.635,.397,.075,.085,MAGENTA)
    txt(ax,.68,.388,"DKAN",7.2,MAGENTA,"bold")
    txt(ax,.724,.448,r"$\alpha_i$",5.7,MAGENTA,"bold")
    txt(ax,.68,.515,r"$\theta_g$",6.5,MAGENTA,"bold")
    lock(ax,.704,.515,MAGENTA,.72)

    # Basis pair enters the blend, not the FV update.
    rounded(ax,.63,.19,.25,.105,"white",GOLD,1.1,.015)
    wave(ax,.65,.213,.065,.055,"smooth",BLUE,1.55)
    wave(ax,.745,.213,.065,.055,"jump",RED,1.55)
    txt(ax,.73,.205,r"$(1-\beta_i)q^{MC}+\beta_iq^{jump}$",6.9,GOLD,"bold")
    arr(ax,(.755,.295),(.83,.37),GOLD,1.25,10,rad=-.18)
    txt(ax,.765,.485,r"$\beta_i=s_i\alpha_i$",6.4,MAGENTA,"bold")

    # The learned component is restricted to reconstruction inside a parent cell.
    ax.plot([.63,.88],[.325,.325],color=GOLD,lw=1.0,zorder=4)
    ax.plot([.63,.63],[.319,.331],color=GOLD,lw=1.0,zorder=4)
    ax.plot([.88,.88],[.319,.331],color=GOLD,lw=1.0,zorder=4)
    txt(ax,.755,.313,"within-cell only",5.7,GOLD,"bold")

    # Trust filters and accepted/fallback paths.
    arr(ax,(.88,.45),(.902,.45),GREEN,1.45,11)
    shield(ax,.919,.45,r"$[q_-,q_+]$",6.0)
    shield(ax,.960,.45,r"$\rho,p>0$",5.8)
    txt(ax,.94,.535,"a posteriori",5.6,GREEN,"bold")
    arr(ax,(.919,.408),(.919,.325),GREEN,1.35,10)
    shield(ax,.919,.29,r"$TV\leq\tau$",6.2)
    arr(ax,(.943,.29),(.967,.29),GREEN,1.35,10)
    for k,hh in enumerate([.77,.72,.58,.30]):
        ax.add_patch(Rectangle((.935+.013*k,.205),.010,.057*hh,
                               facecolor="#78B47F",edgecolor=GREEN,lw=.55,zorder=8))
    txt(ax,.958,.25,r"$\widetilde{\mathbf{U}}_{i,q}$",5.7,GREEN,"bold")
    txt(ax,.955,.187,"accept",6.2,GREEN,"bold")
    rounded(ax,.89,.105,.075,.06,LIGHT,GREY,1.0,.012)
    wave(ax,.902,.115,.050,.035,"smooth",BLUE,1.25)
    txt(ax,.927,.098,"MC",5.8,GREY,"bold")
    arr(ax,(.919,.252),(.925,.165),RED,1.0,9,rad=.20,ls="--")
    txt(ax,.888,.205,"reject",5.8,RED,"bold")

    # Algebraic invariants: the visual anchor of the online band.
    rounded(ax,.055,.115,.52,.13,"#EEF5FA",BLUE,1.1,.016)
    txt(ax,.315,.188,r"$\bar{\mathbf{U}}^{n+1}_i=\bar{\mathbf{U}}^{n}_i-\frac{\Delta t}{\Delta x}(\widehat{\mathbf{F}}_{i+1/2}-\widehat{\mathbf{F}}_{i-1/2})$",8.4,BLUE,"bold")
    txt(ax,.315,.139,r"$\widehat{\mathbf{F}}_{i+1/2}^{(L)}=\widehat{\mathbf{F}}_{i+1/2}^{(R)}$",7.0,BLUE,"bold")
    rounded(ax,.605,.09,.25,.075,GREEN_LIGHT,GREEN,1.1,.014)
    txt(ax,.73,.127,r"$\frac{1}{Q}\sum_q\widetilde{\mathbf{U}}_{i,q}=\bar{\mathbf{U}}_i$",7.6,GREEN,"bold")

    # Frozen checkpoint link.
    arr(ax,(.665,.745),(.68,.53),VIOLET,1.1,10,rad=.08,ls="--")
    txt(ax,.72,.66,r"freeze $\theta_g$",6.4,VIOLET,"bold")

    save(fig,"fig1_offline_online_cse_dkan_fv")


def fig2_trust_and_evidence():
    fig, ax = canvas((12.2, 5.6))

    # Panel a: successive convex projections toward the fixed monotone profile.
    txt(ax,.035,.94,"a",11,INK,"bold")
    txt(ax,.08,.94,"TRUST PROJECTION",7.5,GREEN,"bold",ha="left")

    rounded(ax,.055,.73,.18,.15,RED_LIGHT,RED,1.35,.018)
    wave(ax,.078,.765,.135,.085,"unsafe",RED,1.9)
    txt(ax,.145,.71,r"$q^{(0)}=q^*_{DKAN}$",7.2,RED,"bold")
    pill(ax,.145,.895,"proposal",RED_LIGHT,RED,.075,.029,5.5)

    # Central projection spine.
    ax.plot([.315,.315],[.76,.28],color="#C7CDD3",lw=2.0,zorder=1)
    ax.plot([.25,.25],[.34,.72],color=BLUE,lw=1.0,ls="--",alpha=.75,zorder=1)
    arr(ax,(.235,.79),(.285,.72),GREEN,1.25,10)
    stages = [
        (.72,r"$\lambda_s$",r"$q\in[q_-,q_+]$",GREEN),
        (.54,r"$\lambda_+$",r"$\rho,p>0$",GREEN),
        (.36,r"$\lambda_{TV}$",r"$TV\leq\tau$",GREEN),
    ]
    kinds = ["filtered1","filtered2","blend"]
    for (y,lam,lab,col),kind in zip(stages,kinds):
        arr(ax,(.25,y),(.285,y),BLUE,1.0,9)
        ax.add_patch(Circle((.315,y),.027,facecolor=GREEN_LIGHT,edgecolor=GREEN,lw=1.4,zorder=7))
        txt(ax,.315,y,lam,7.0,GREEN,"bold")
        arr(ax,(.342,y),(.395,y),GREEN,1.25,10)
        rounded(ax,.395,y-.065,.165,.13,"white",GREEN,1.1,.014)
        wave(ax,.415,y-.032,.125,.070,kind,GREEN if kind=="blend" else MAGENTA,1.65)
        txt(ax,.477,y-.078,lab,6.4,GREEN,"bold")
        if y > .4:
            arr(ax,(.315,y-.027),(.315,y-.153),GREEN,1.15,9)

    for x,y,lab in [(.477,.804,"stencil projection"),(.477,.624,"positivity bisection"),
                    (.477,.444,"TV trust")]:
        pill(ax,x,y,lab,"white",GREEN,.125,.028,5.4)

    txt(ax,.17,.625,r"$q^{(k+1)}=q^{MC}+\lambda_k(q^{(k)}-q^{MC})$",7.2,GREY,"bold")
    wave(ax,.105,.54,.105,.075,"smooth",BLUE,1.6)
    txt(ax,.157,.515,r"fixed $q^{MC}$",6.4,BLUE,"bold")
    arr(ax,(.21,.575),(.25,.54),BLUE,1.0,9,rad=-.10)
    txt(ax,.235,.327,r"$\lambda_k\in[0,1]$",5.6,BLUE,"bold")

    # Accept or fallback without changing the cell mean.
    arr(ax,(.477,.295),(.477,.22),GREEN,1.3,10)
    arr(ax,(.477,.295),(.285,.20),RED,1.0,9,rad=.15,ls="--")
    rounded(ax,.395,.085,.18,.12,GREEN_LIGHT,GREEN,1.25,.016)
    wave(ax,.417,.115,.135,.065,"blend",GREEN,1.8)
    txt(ax,.485,.075,"accept",6.5,GREEN,"bold")
    ax.plot([.535,.545,.562],[.168,.155,.181],color=GREEN,lw=1.8,
            solid_capstyle="round",solid_joinstyle="round",zorder=10)
    rounded(ax,.17,.085,.15,.12,LIGHT,GREY,1.15,.016)
    wave(ax,.19,.115,.11,.065,"smooth",BLUE,1.6)
    txt(ax,.245,.075,"fallback",6.5,GREY,"bold")
    ax.add_patch(Arc((.19,.17),.030,.028,theta1=35,theta2=325,
                     color=GREY,lw=1.25,zorder=9))
    arr(ax,(.177,.164),(.181,.153),GREY,.9,7,rad=.05)
    txt(ax,.13,.215,"reject",6.2,RED,"bold")
    rounded(ax,.06,.30,.17,.095,"white",BLUE,1.0,.014)
    txt(ax,.145,.347,r"$\sum_q(q^{(k)}_{i,q}-\bar q_i)=0$",7.0,BLUE,"bold")
    lock(ax,.213,.347,BLUE,.82)
    arr(ax,(.23,.347),(.25,.347),BLUE,1.0,8)
    txt(ax,.145,.287,"exact mean",5.6,BLUE,"bold")

    # Divider and panel b: what is guaranteed, enforced, or merely audited.
    ax.plot([.62,.62],[.075,.92],color="#D8DDE2",lw=1.2,zorder=1)
    txt(ax,.655,.94,"b",11,INK,"bold")
    txt(ax,.70,.94,"EVIDENCE BOUNDARY",7.5,GREY,"bold",ha="left")

    rounded(ax,.665,.51,.30,.36,"white",INK,1.0,.022)
    txt(ax,.68,.845,"STRUCTURAL",6.7,BLUE,"bold",ha="left")
    rounded(ax,.69,.69,.25,.115,BLUE_LIGHT,BLUE,1.25,.015)
    pill(ax,.902,.79,"by construction",BLUE_LIGHT,BLUE,.09,.026,5.1)
    ax.add_patch(Circle((.72,.748),.019,facecolor="white",edgecolor=BLUE,lw=1.1,zorder=7))
    txt(ax,.72,.748,"=",8.5,BLUE,"bold")
    txt(ax,.755,.766,r"$\widehat F^{L}=\widehat F^{R}$",7.2,BLUE,"bold",ha="left")
    txt(ax,.755,.726,r"$Q^{-1}\sum_q\widetilde U_q=\bar U$",7.2,BLUE,"bold",ha="left")
    lock(ax,.919,.748,BLUE,.9)

    txt(ax,.68,.65,"HARD FILTERS",6.7,GREEN,"bold",ha="left")
    rounded(ax,.69,.535,.25,.095,GREEN_LIGHT,GREEN,1.25,.015)
    pill(ax,.902,.642,"a posteriori",GREEN_LIGHT,GREEN,.08,.026,5.1)
    shield(ax,.724,.582,r"$q_\pm$",5.6)
    shield(ax,.813,.582,r"$\rho,p$",5.6)
    shield(ax,.902,.582,r"$TV$",5.8)

    # Audit layer is intentionally outside the guarantee box.
    audit_box = rounded(ax,.665,.125,.30,.26,GOLD_LIGHT,GOLD,1.25,.020)
    audit_box.set_linestyle((0,(4,2)))
    txt(ax,.68,.355,"AUDIT",6.7,GOLD,"bold",ha="left")
    pill(ax,.91,.355,"diagnostic only",GOLD_LIGHT,GOLD,.095,.026,5.1)
    items = [
        (.715,.28,"S",r"$\Delta S\geq0$"),
        (.84,.28,"RH",r"$\mathcal{R}_{RH}$"),
        (.715,.19,"w",r"$w_{shock}$"),
        (.84,.19,"t",r"$t_{CPU}$"),
    ]
    for x,y,glyph,s in items:
        ax.add_patch(Circle((x,y),.025,facecolor="white",edgecolor=GOLD,lw=1.1,zorder=7))
        txt(ax,x,y,glyph,5.8,GOLD,"bold")
        txt(ax,x+.04,y,s,7.0,GOLD,"bold",ha="left")
    ax.add_patch(Circle((.925,.255),.019,facecolor="white",edgecolor=GOLD,lw=1.1,zorder=7))
    ax.add_patch(Circle((.925,.255),.008,facecolor=GOLD_LIGHT,edgecolor=GOLD,lw=.9,zorder=8))
    ax.plot([.938,.955],[.238,.22],color=GOLD,lw=1.4,zorder=8)

    # Small semantic key, no paragraph legend.
    txt(ax,.665,.075,"guarantee",6.2,BLUE,"bold",ha="left")
    txt(ax,.765,.075,"enforce",6.2,GREEN,"bold",ha="left")
    txt(ax,.855,.075,"measure",6.2,GOLD,"bold",ha="left")

    save(fig,"fig2_constraint_evidence_logic")


if __name__ == "__main__":
    fig1_train_then_deploy()
    fig2_trust_and_evidence()
    print("wrote two flow-and-constraint schematics")
