from account import SavingsAccount, CheckingAccount
from customer import Customer
from bank import Bank


def main():
    # Create a bank
    my_bank = Bank("My Bank")

    # Add customers
    customer1 = Customer("John Doe", "001")
    customer2 = Customer("Jane Doe", "002")

    my_bank.add_customer(customer1)
    my_bank.add_customer(customer2)

    # Create accounts for customers
    acc1 = SavingsAccount("S001", balance=1000.0)
    acc2 = CheckingAccount("C001", balance=500.0)

    customer1.add_account(acc1)
    customer1.add_account(acc2)

    # Perform some operations
    acc1.deposit(500.0)
    acc2.withdraw(100.0)

    # Apply interest
    acc1.apply_interest()

    # Display all banks, customers, and accounts
    Bank.display_all()


if __name__ == "__main__":
    main()
