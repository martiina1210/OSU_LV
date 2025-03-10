dict = dict()

fhand = open("song.txt")
for line in fhand:
    line = line.strip("\n")
    words = line.split(" ")
    if words == "":
        pass
    for word in words:
        if word in dict:
            dict[word]+=1
        else:
            dict[word]=1
temp = list()
for key in dict:
    if dict[key]==1:
        temp.append(key)
print("Ukupno ima "+str(len(temp))+" ključeva s vrjednošću 1")
for key in temp:
    print(key)
