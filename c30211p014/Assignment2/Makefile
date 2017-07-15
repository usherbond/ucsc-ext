# Minimal make
#
RMD = Assignment2.Rmd
HTML = $(RMD:%.Rmd=%.html)

all : $(HTML)

R_OPTS=--no-save --no-restore --no-init-file --no-site-file
$(HTML) : $(RMD)
	R ${R_OPTS} -e "rmarkdown::render('Assignment2.Rmd')"

.PHONY: open clean
open : $(HTML)
	open $(HTML)

clean : 
	rm $(HTML)
