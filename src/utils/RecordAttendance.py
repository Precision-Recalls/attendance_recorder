import time


def takeAttendance(name):
    # It will get the current date
    today = time.strftime('%d_%m_%Y')
    # To create a file if it doesn't exists
    f = open(f'output/attendance_sheet_{today}.csv', 'a')
    f.close()

    exclude_names = []

    # It will read the CSV file and check if the name
    # is already present there or not.
    # If the name doesn't exist there, it'll be added
    # to a list called 'names'
    with open(f'output/attendance_sheet_{today}.csv', 'r') as f:
        data = f.readlines()
        names = []
        for line in data:
            entry = line.split(',')
            names.append(entry[0])

    # It will check it the name is in the list 'names'
    # or not. If not then, the name will be added to
    # the CSV file along with the entering time
    with open(f'output/attendance_sheet_{today}.csv', 'a') as fs:
        if name not in names:
            current_time = time.strftime('%H:%M:%S')
            if name not in exclude_names:
                fs.write(f"\n{name}, {current_time}")
    return today

