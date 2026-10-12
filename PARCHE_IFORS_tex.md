# Parche LaTeX — IFORS 2026 (aplicar el relato)

Reemplazos puntuales sobre `IFORS2026_Araneda_revised.tex`. Aplican el feedback
del profesor: revelar a SMAC-SK como protagonista y reconciliar el número.

---

## 1. Objective — anunciar al protagonista

**Reemplazar** el frame Objective por:

```latex
\begin{frame}{Objective}
	\justifying
	Design, formulate, and evaluate a prescriptive decision-support framework for
	capacity planning in Minimally Invasive Gynecology.
	\vspace{1em}

	\begin{tcolorbox}[colback=okgreen!8, colframe=okgreen, boxrule=0.8pt, arc=2mm]
		\textbf{Given the methodological gap, the proposed instrument is
		\textcolor{okgreen}{SMAC-SK}} (SMAC with a Stochastic-Kriging surrogate):
		it reuses an off-the-shelf, easy-to-implement structure \emph{and} handles
		the heteroscedastic, non-smooth noise that breaks the alternatives.
	\end{tcolorbox}
\end{frame}
```

---

## 2. Proposed Framework — nombrar el bloque central SMAC-SK

En el `tikzpicture`, **reemplazar** el nodo `dfo`:

```latex
\node[block, right=of des, fill=okgreen!12, draw=okgreen] (dfo) {%
  \textbf{SMAC-SK}\\[4pt]
};
```

y la etiqueta inferior:

```latex
\node[label, below=0.7cm of dfo]  {\textit{Core contribution}\\(recommended method)};
```

(Opcional) actualizar el pie del benchmark a tus números reales:

```latex
{\scriptsize Benchmark: 7 algorithms $\times$ 15 seeds, $\sim$150 evaluations,
Common Random Numbers (CRN); incumbent re-evaluated $n=50$.}
```

> Nota: el `.tex` actual dice "7 × 10 seeds, 18 evaluations". Tus corridas son
> **15 seeds** y **~150 evaluaciones** por corrida (las 18 réplicas son del
> Appendix A, otra cosa). Conviene unificar.

---

## 3. Results: Optimized vs. Manual — usar la figura con SMAC-SK en verde

La figura `fig_pareto.png` ya viene con SMAC-SK como estrella verde. Solo
añade un subtítulo que dirija la mirada:

```latex
\begin{frame}{Results: Optimized vs.\ Manual}
	\begin{figure}
		\centering
		\includegraphics[width=1\textwidth]{fig_pareto.png}
		\caption{\small TTS vs.\ throughput. \textcolor{okgreen}{SMAC-SK (green star)}
		is the recommended method; all algorithms Pareto-dominate the manual options.}
	\end{figure}
\end{frame}
```

---

## 4. Convergence — dirigir a la curva verde

```latex
\begin{frame}{Algorithm convergence (sample efficiency)}
	\begin{figure}
		\centering
		\includegraphics[width=1\textwidth]{conv.png}
		\caption{\small \textcolor{okgreen}{Green = SMAC-SK.} Compared with SMAC
		without SK, the SK term — not luck — drives the gain.}
	\end{figure}
\end{frame}
```

(Usa la nueva `fig_convergencia.png` renombrada a `conv.png`: SMAC-SK en verde grueso.)

---

## 5. Discussion — reconciliar 164 → 177 d

**Reemplazar** la primera viñeta del frame Discussion/Results:

```latex
\item \textbf{Main result:} \textcolor{okgreen}{SMAC-SK} reduced TTS from
\textbf{263 to 177 days} ($-33\%$) while patients served increased from
\textbf{1{,}148 to $\sim$2{,}180} ($+90\%$).
```

> ⚠ El deck decía **164 días**, pero la re-evaluación honesta ($n=50$) da
> **174–177 d** para el clúster top (SMAC-SK = 177, SPSA = 174, empate
> estadístico). Titular con los números de **SMAC-SK (177 d)** es coherente con
> que él es el protagonista. Si tienes una fuente para 164, verifícala antes.

---

## 6. (Opcional) Appendix F — marcar SMAC-SK

En la tabla del Appendix F, resaltar la fila SMAC-SK:

```latex
\rowcolor{okgreen!12}
SMAC-SK $\bigstar$ & Stochastic Kriging & Mixed (int+cont) & Part. & \xmark & Asymptotic$^*$ & Lindauer (2022); Ankenman (2010) \\
```

(requiere `\usepackage[table]{xcolor}`, ya cargado, y `\usepackage{bbding}` o
usar `$\star$` para la estrella).
