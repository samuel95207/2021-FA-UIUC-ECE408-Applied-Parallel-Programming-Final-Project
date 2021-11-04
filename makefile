FOLDER = .
all:
	./rai -p $(FOLDER) --queue rai_amd64_ece408
	
submit_m1:
	cp ./docs/m1/report.pdf .
	./rai -p $(FOLDER) --queue rai_amd64_ece408 --submit=m1
	rm report.pdf
	
submit_m2:
	cp ./docs/m1/report.pdf .
	./rai -p $(FOLDER) --queue rai_amd64_ece408 --submit=m2	
	rm report.pdf

submit_m3:
	cp ./docs/m1/report.pdf .
	./rai -p $(FOLDER) --queue rai_amd64_ece408 --submit=m2	
	rm report.pdf

history:
	./rai -p $(FOLDER) history