from account import Account
from customer import Customer
class Bank:
    all_banks = []

    def __init__(self, name):
        self.name = name
        self.customers = {}
        Bank.all_banks.append(self)

    def add_customer(self, customer):
        self.customers[customer.customer_id] = customer

    def get_customer(self, customer_id):
        return self.customers.get(customer_id)

    @staticmethod
    def display_all():
        print("All Banks:")
        for bank in Bank.all_banks:
            print(f"Bank Name: {bank.name}")
            print("Customers:")
            for customer_id, customer in bank.customers.items():
                print(f"  Customer ID: {customer_id}, Name: {customer.name}")
                print("  Accounts:")
                for acc_number, account in customer.accounts.items():
                    print(f"    Account Number: {acc_number}, Balance: {account.get_balance()}")
        print()

        print("All Accounts:")
        for account in Account.all_accounts:
            print(f"Account Number: {account.account_number}, Balance: {account.get_balance()}")
        print()

        print("All Customers:")
        for customer in Customer.all_customers:
            print(f"Customer ID: {customer.customer_id}, Name: {customer.name}")
        print()
