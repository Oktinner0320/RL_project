// MIT License
//
// Copyright (c) 2023-2025 Junliang HU <jlhu@cse.cuhk.edu.hk>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Derived from clean-acmart 0.0.1; see SOURCES.md and vendor/ for the original.
// 基于固定版本的本地改编；保留标题、作者、编号和引用布局，替换课程排版。

// The four metadata arguments and body are explicit; this file owns formatting.
// 四项元数据与正文由 main.typ 提供；排版参数集中在本文件。
#let course-acm(
  title, authors, abstract, keywords, body,
  lang: "en", fonts: ("Arial",),
  abstract-label: [Abstract], keywords-label: [Keywords:],
  ink-padding: 0pt,
) = {
  set document(title: title, author: authors, keywords: keywords)
  set page(paper: "a4", margin: 1in, columns: 1, numbering: "1")

  // Keep an 18pt baseline while reserving space for glyph descenders.
  // 普通正文：12pt 上缘 + 侧承预留 + (6pt - 预留) = 18pt 基线间距。
  set text(
    font: fonts, size: 12pt, lang: lang,
    top-edge: 1em, bottom-edge: -ink-padding,
    // Keep punctuation inside the measured one-inch text boundary.
    // 关闭标点悬挂，确保可见字符也位于一英寸正文边界内。
    overhang: false,
  )
  set par(leading: 6pt - ink-padding, spacing: 6pt - ink-padding, justify: true, first-line-indent: 1em)
  show par: it => if ink-padding > 0pt { pad(x: ink-padding, it) } else { it }
  set block(spacing: 6pt - ink-padding)
  set heading(numbering: "1.1")
  show heading: set text(font: fonts, size: 12pt, weight: "bold")
  show heading: set block(above: 12pt, below: 6pt - ink-padding, sticky: true)
  show raw: set text(font: fonts, size: 12pt)
  show math.equation: set text(font: "Cambria Math", size: 12pt)
  set math.equation(numbering: "(1)")
  set figure(gap: 6pt - ink-padding)
  show figure.caption: set text(size: 10pt)
  show figure: it => if ink-padding > 0pt { pad(x: ink-padding, y: 0.5pt, it) } else { it }
  set place(clearance: 12pt)
  set enum(indent: 12pt, body-indent: 6pt)
  set list(indent: 12pt, body-indent: 6pt)

  // References begin on their own page and keep the same text typography.
  // 文献另起页，字号和行距不沿用上游的缩小规则。
  show bibliography: it => {
    pagebreak()
    set text(font: fonts, size: 12pt)
    set par(leading: 6pt - ink-padding, spacing: 6pt - ink-padding, first-line-indent: 0pt)
    pad(x: ink-padding, it)
  }

  // In-flow title avoids the two-column floating title used by the upstream.
  // 单栏标题按正常文档流排版，避免浮动标题与摘要重叠。
  // Keep glyph side bearings inside the page margin without scaling text.
  // 中文稿可预留微量字形侧承空间；英文默认不增加内垫。
  align(center, block[
    #strong(title)
    #v(6pt)
    #authors.join(linebreak())
  ])
  block(above: 12pt, below: 6pt)[#strong(abstract-label)]
  abstract
  block(above: 6pt, below: 12pt)[#strong(keywords-label) #keywords.join(if lang == "zh" { "；" } else { ", " })]
  body
}
