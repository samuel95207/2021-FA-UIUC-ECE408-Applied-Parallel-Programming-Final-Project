FOLDER = src
all:
	./rai -p $(FOLDER) --queue rai_amd64_ece408

exclusive:
	./rai -p $(FOLDER) --queue rai_amd64_exclusive
	
submit_m1:
	cp ./docs/m1/report.pdf $(FOLDER)
	./rai -p $(FOLDER) --queue rai_amd64_ece408 --submit=m1
	rm report.pdf
	
submit_m2:
	cp ./docs/m2/report.pdf $(FOLDER)
	./rai -p $(FOLDER) --queue rai_amd64_ece408 --submit=m2	
	rm report.pdf

submit_m3:
	cp ./docs/m3/report.pdf $(FOLDER)
	./rai -p $(FOLDER) --queue rai_amd64_ece408 --submit=m2	
	rm report.pdf

history:
	./rai -p $(FOLDER) history