import string

arr = [66,2,3,67]

for i in range(len(arr)):
    print(arr[i])




my_string = "blah, lots , of , spaces-something, AGSDere_something.?"

my_string = my_string.replace('_', ' ').lower()
my_string = my_string.replace('-', ' ')

my_string = my_string.translate(str.maketrans('', '', string.punctuation))


print(my_string.split())

# result = [x.strip() for x in my_string.split(',')]
#
# arr = []
# for i in result:
#     arr.extend(i.split('_'))
#
# print(arr)