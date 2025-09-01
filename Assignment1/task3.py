
#####Task 3#####

products = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
totalSum = 0

print("Supermarket")
print("====================")

while True:
    userInput = int(input("Please select product (1-10) 0 to Quit: "))
    if userInput == 0:
        break
    elif 1 <= userInput <= 10:
        productPrice = products[userInput-1]
        totalSum += productPrice
        print(f"Product: {userInput} Price: {productPrice}")
    else:
        print("Invalid product number.")

print(f"Total: {totalSum}")

paymentTaken =  int(input("Payment: "))

changePayment = paymentTaken - totalSum

print(f"Change: {changePayment}")
