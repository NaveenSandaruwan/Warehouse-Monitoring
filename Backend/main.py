import Backend.mysqlApi as mysqlApi
import Backend.mongoDBapi as mongoDBapi

def main():
    while True:
        print("Add a worker: press 1")
        print("Get a worker: press 2")
        print("Exit: press 3")
        choice = input("Enter your choice: ")
        if choice == '1':
            name = input("Enter worker's name: ")
            mysqlApi.add_worker(name)
        elif choice == '2':
            worker_id = input("Enter worker's ID: ")
            worker = mysqlApi.get_worker(worker_id)
            if not worker:
                print("Worker not found")
            print(worker)
        elif choice == '3':
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == '__main__':
    main()