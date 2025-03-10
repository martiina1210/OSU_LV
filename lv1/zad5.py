fhand = open("SMSSpamCollection.txt")

ham = list()
spam = list()

hamCounter = 0
hamAverage = 0

spamCounter = 0
spamAverage = 0
endswithCounter =0

for line in fhand:
    if line.startswith("ham"):
        line = line.strip("ham")
        line = line.rstrip()
        line = line.strip("\n")
        line = line.split(" ")
        hamCounter+=1
        hamAverage+=len(line)
        ham.append(line)
    else:
        line = line.strip("spam")
        line = line.rstrip()
        line = line.strip("\n")
        if line.endswith("!"):
            endswithCounter+=1
        line = line.split(" ")
        spamCounter+=1
        spamAverage+=len(line)
        spam.append(line)

print("Prosječan broj riječi u ham: "+str(round(hamAverage/hamCounter))+"\n")
print("Prosječan broj riječi u spam: "+str(round(spamAverage/spamCounter))+"\n")
print("Broj spam poruka koje zavrsavaju s !: "+str(endswithCounter))
