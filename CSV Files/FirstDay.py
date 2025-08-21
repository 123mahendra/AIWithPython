######AI with python###############


####Exercise 1##########
###Type 1

citizenship = input("Do you have a citizenship? Type yes or no:").lower()

if citizenship == "yes":
    age = int(input("Enter your age:"))
    if age >= 18:
        print("Congratulations, you can vote!")
    else:
        print(f"Sorry, you can't vote! you have to wait more {18 - age} years.")
else:
    print("Sorry, you can't vote!")

###Type 2

citizenship = input("Do you have a citizenship? Type yes or no:").lower()

age = int(input("Enter your age:"))

if citizenship == 'yes' and age >= 18:
    print("Congratulations, you can vote!")
elif citizenship == 'yes' and age < 18:
    print(f"Sorry, you can't vote! you have to wait more {18 - age} years.")
else:
    print("Sorry, you can't vote!")

#########Exercise 2#######

age = int(input("Enter your age: "))

if age >= 18:
    print("You can take medicine!")
elif age >= 15 and age < 18:
    weight = int(input("Enter your weight: "))
    if weight >= 55:
        print("You can take medicine!")
    else:
        print("Sorry, you can not take medicine.")
else:
    print("Sorry, you can not take medicine.")


######Exercise 3###########

months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

seasonWithMonths = {
    "January": "Winter",
    "February":"Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Autumn",
    "October": "Autumn",
    "November": "Autumn",
    "December": "Winter"
}

month = int(input("Enter the month: "))

if month in months:
    monthName = months[month]
    print(f"Month is {monthName}")
    print(f"Season is {seasonWithMonths[monthName]}")
else:
    print("You have to input 1-12 value.")

#########Exercise 4#########

balance = 1000

while balance >= 0:
    Type = input("Enter type as deposit, withdraw, or check balance:").lower()

    if(Type == "deposit"):
        newBalance = int(input("Enter amount to deposit: "))
        balance = balance + newBalance
        print(f"Successfully deposited {newBalance}, Now your balance is:", balance)
    elif(Type == "withdraw"):
        newBalance = int(input("Enter amount to withdraw: "))
        if(balance >= newBalance):
            balance = balance - newBalance
            print(f"Successfully withdraw {newBalance}, Now your balance is:", balance)
        else:
            print("You don't have enough money!")
    elif(Type == "check balance"):
        print(f"Your balance is {balance}")
    else:
        break
else:
    print("Insufficient balance")


############Exercise 5###########

x = int(input("Enter a number: "))
y = int(input("Enter another number: "))

number = int(input("Enter 1 for add, 2 for subtract, 3 for multiply, 4 for divide:"))

def calculate(number,x,y):
    if number == 1:
        return x + y
    elif number == 2:
        return x - y
    elif number == 3:
        return x * y
    elif number == 4:
        return x / y

calculation = calculate(number,x,y)

print(calculation)


##########Exercise 6###########

number = int(input("Enter a number: "))

for i in range(1,11):
    print(f"{number} * {i} = {i*number}")
