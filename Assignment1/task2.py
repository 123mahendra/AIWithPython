
#####Task 2#####

productList = []

while True:
    userInput = input("Would you like to (1)Add or (2)Remove items or (3)Quit?: ")
    if userInput == "1":
        product = input("What will be added?: ")
        productList.append(product)
    elif userInput == "2":
        if len(productList) == 0:
            print("The product list is empty!")
            continue

        print(f"There are {len(productList)} items in the list.")
        indexNumber = int(input("Which item is deleted?: "))
        if indexNumber < len(productList):
            del productList[indexNumber]
        else:
            print("Incorrect selection.")
    elif userInput == "3":
        print("The following items remain in the list:")
        for i in productList:
            print(i)
        break
    else:
        print("Incorrect selection.")



