def total_euro(hours,rate):
    return hours*rate
hours=float(input("Broj sati: "))
rate=float(input("Unesite zaradu: "))

print(f"Money earned: {total_euro(hours,rate)}")