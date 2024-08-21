class Customer:
    all_customers = []

    def __init__(self, name, customer_id):
        self.name = name
        self.customer_id = customer_id
        self.accounts = {}
        Customer.all_customers.append(self)

    def add_account(self, account):
        self.accounts[account.account_number] = account

    def get_account(self, account_number):
        return self.accounts.get(account_number)

    def get_total_balance(self):
        return sum(account.get_balance() for account in self.accounts.values())
