def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

lista = list()

while True:
    temp = input()
    if isfloat(temp) or temp.isdigit():
        lista.append(float(temp))
    else:
        if temp != "Done":
            print("Unos mora biti broj")
        else:
            break

len = len(lista)
average = sum(lista)/len
min = min(lista)
max = max(lista)

print("Korisnik je unio: "+str(len)+" brojeva\n")
print("Srednja vrijednost elemenata liste: "+str(average))
print("Minimalna vrijednost: "+str(min))
print("Maksimalna vrijednost: "+str(max))
