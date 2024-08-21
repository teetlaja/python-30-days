import argparse
import json
import os
import sys

# Default path to the JSON file that stores user data
DEFAULT_DB_FILE = 'users.json'


def load_users(db_file):
    """Load users from the JSON file."""
    if os.path.exists(db_file):
        with open(db_file, 'r') as file:
            return json.load(file)
    return []


def save_users(users, db_file):
    """Save list of users to the JSON file."""
    with open(db_file, 'w') as file:
        json.dump(users, file, indent=4)


def add_user(args):
    """Add a new user to the database."""
    users = load_users(args.db_file)
    user = {
        'id': len(users) + 1,
        'name': args.name,
        'email': args.email
    }
    users.append(user)
    save_users(users, args.db_file)
    print(f"User {args.name} added.")


def search_user(args):
    """Search for users by name."""
    users = load_users(args.db_file)
    results = [user for user in users if args.name.lower() in user['name'].lower()]
    if results:
        for user in results:
            print(f"ID: {user['id']}, Name: {user['name']}, Email: {user['email']}")
    else:
        print("No users found.")


def update_user(args):
    """Update a user's email by ID."""
    users = load_users(args.db_file)
    user = next((user for user in users if user['id'] == args.id), None)
    if user:
        user['email'] = args.email
        save_users(users, args.db_file)
        print(f"User ID {args.id} updated.")
    else:
        print(f"No user found with ID {args.id}.")


def delete_user(args):
    """Delete a user by ID."""
    users = load_users(args.db_file)
    users = [user for user in users if user['id'] != args.id]
    save_users(users, args.db_file)
    print(f"User ID {args.id} deleted.")


def main():
    parser = argparse.ArgumentParser(description="User Manager CLI Tool")
    parser.add_argument('--db-file', type=str, default=DEFAULT_DB_FILE,
                        help='Path to the JSON file that stores user data')
    subparsers = parser.add_subparsers(help="Available commands")

    # Add user command
    parser_add = subparsers.add_parser('add', help="Add a new user")
    parser_add.add_argument('name', type=str, help="Name of the user")
    parser_add.add_argument('email', type=str, help="Email of the user")
    parser_add.set_defaults(func=add_user)

    # Search user command
    parser_search = subparsers.add_parser('search', help="Search users by name")
    parser_search.add_argument('name', type=str, help="Name to search for")
    parser_search.set_defaults(func=search_user)

    # Update user command
    parser_update = subparsers.add_parser('update', help="Update user email by ID")
    parser_update.add_argument('id', type=int, help="User ID")
    parser_update.add_argument('email', type=str, help="New email of the user")
    parser_update.set_defaults(func=update_user)

    # Delete user command
    parser_delete = subparsers.add_parser('delete', help="Delete user by ID")
    parser_delete.add_argument('id', type=int, help="User ID")
    parser_delete.set_defaults(func=delete_user)

    # Parse and execute the command
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
