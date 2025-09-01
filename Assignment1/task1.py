
#####Task 1#####

def tester(string="Too short"):
    print(string)

def main():
    while True:
        userInput = input("Write something (quit ends): ")
        if userInput == "quit":
            break
        if len(userInput) >= 10:
            tester(userInput)
        else:
            tester()

main()