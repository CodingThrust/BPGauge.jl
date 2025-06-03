using BPGauge
using Documenter

DocMeta.setdocmeta!(BPGauge, :DocTestSetup, :(using BPGauge); recursive=true)

makedocs(;
    modules=[BPGauge],
    authors="ArrogantGao <xz.gao@connect.ust.hk> and contributors",
    sitename="BPGauge.jl",
    format=Documenter.HTML(;
        canonical="https://ArrogantGao.github.io/BPGauge.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ArrogantGao/BPGauge.jl",
    devbranch="main",
)
