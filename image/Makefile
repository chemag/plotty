


all: none

data.png:
	../plotty-plot.py --inplot plots.pbtxt \
        --lineid simple/data \
        $@


lines.png:
	../plotty-plot.py --inplot plots.pbtxt \
        --lineid simple/dot \
        --lineid simple/horizontal \
        --lineid simple/vertical \
        --lineid simple/diagonal \
        --lineid simple/line \
        $@

xtickstxt.png:
	../plotty-plot.py --inplot plots.pbtxt \
        --lineid xtickscol/data \
        $@
