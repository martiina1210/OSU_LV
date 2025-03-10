try:
    grade = input("Unesite broj u intervalu [0.0,1.0]: ")
    grade = float(grade)
    if grade < 0.0 or grade > 1.0:
        raise Exception("Broj mora biti u intervalu [0.0, 1.0]!")
    if grade >=0.9:
        print("A")
    elif grade >=0.8:
        print("B")
    elif grade >=0.7:
        print("C")
    elif grade >=0.6:
        print("D")
    else:
        print("F")

except Exception as ex:
    print("Doslo je do greske."+str(ex))
