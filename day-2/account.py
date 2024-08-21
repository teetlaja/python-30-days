class Account:
    all_accounts = []

    def __init__(self, account_number, balance=0.0):
        self.account_number = account_number
        self.balance = balance
        Account.all_accounts.append(self)

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
        else:
            print("Invalid deposit amount")

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
        else:
            print("Invalid or insufficient funds for withdrawal")

    def get_balance(self):
        return self.balance


class SavingsAccount(Account):
    def __init__(self, account_number, balance=0.0, interest_rate=0.01):
        super().__init__(account_number, balance)
        self.interest_rate = interest_rate

    def apply_interest(self):
        self.balance += self.balance * self.interest_rate


class CheckingAccount(Account):
    def __init__(self, account_number, balance=0.0, overdraft_limit=100.0):
        super().__init__(account_number, balance)
        self.overdraft_limit = overdraft_limit

    def withdraw(self, amount):
        if 0 < amount <= self.balance + self.overdraft_limit:
            self.balance -= amount
        else:
            print("Invalid or insufficient funds for withdrawal with overdraft limit")
