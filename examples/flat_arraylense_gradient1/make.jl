#=

```shell
julia make.jl
jupyter nbconvert --to notebook --execute --inplace example.ipynb
```

The first line converts src.jl to a notebook.
The second line runs the notebook.

If you want to save that in another format do the following 

jupyter nbconvert --to html     example.ipynb --output example.html
jupyter nbconvert --to markdown example.ipynb --output example.md
jupyter nbconvert --to webpdf   example.ipynb --output example.pdf
```

=#

using Literate                 

config = Dict(                 
    "name"          => "example",
    "documenter"    => false,
    "keep_comments" => false,  
    "execute"       => false, 
    "credit"        => false,  
) 

Literate.notebook("src.jl"; config=config)                              
