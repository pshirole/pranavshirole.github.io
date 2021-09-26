---
layout: post
title: Python Basics - Part 1
tags: [python]
---

This is Part 1 of a Python tutorial for beginners.

As a self-taught Data Scientist and programmer, I always get asked about how I started my path towards learning, and a lot of non-coders ask me about how they can learn more about Data Science. And while I tell them about the umpteen Data Analytics and Data Visualization tools, various Machine Learning algorithms, and which Deep Learning frameworks to choose, it all starts with learning Python. Python is an interpreted high-level general-purpose programming language. In my view, Python is the best programming language to learn, in order to become a Data Scientist, owing to its readability, non-complexity, its large standard libraries, and its huge community.  
I have been the beneficiary of several of books and YouTube tutorials that have helped me become a better Python Developer. This blog post is my way of giving back to the community. This might not be the best place to start learning how to code in Python. However, this blog post aims to be a good cheat sheet for beginners trying to look something up or if they want a refresher as to how certain objects perform.

---
### Index 

[Display Output](#display-output)  
[Getting information from the user](#getting-information-from-the-user)  
[Comments](#comments)  
[String concepts](#string-concepts)  
[Working with Numbers](#working-with-numbers)  
[Working with Dates](#working-with-dates)  
[Error Handling](#error-handling)  
[Handling Conditions](#handling-conditions)  
[Collections](#collections)  
[Random Module](#random-module)  
[Loops](#loops)    
[Functions](#functions)  


---
### Display Output
You can use the `print` function to display output to your console. You can use either single or double quotes; just make sure that you stick to one for consistency.


```python
print('Hello, World!')
print("Python is great")
```

    Hello, World!
    Python is great


#### Displaying blank lines 
Blank lines make the output more readable. For a blank line, you can insert a `print` function with nothing inside. Each `print` function prints on a new line by default.  
You can also use `\n` (the newline character) to print a new line at the end of a string or right in the middle of a string.


```python
print('Hello')
print()
print('Above is a blank line\n')
print('Blank line \nin the middle of a string')
```

    Hello
    
    Above is a blank line
    
    Blank line 
    in the middle of a string


---
### Getting information from the user
You can use the `input` to ask for information from your user. We pass in a message with the `input` function.


```python
name = input('What is your name? ')
print(name)
```

    What is your name? Pranav
    Pranav


Here, whatever value is typed in by the user will be stored in the variable `name` and can be used as needed. We've chosen to print the value of `name` on the screen.

---
### Comments
Comments are a way of documenting your code. Comments can be added using `#`. These lines of code will not execute.


```python
# print('Hello')
# the above code of line will not execute
# but the below one will 
print('How are you?')
```

    How are you?


You can also use `'''  '''` for multi-line comments.


```python
'''
Python is an important
programming language in
your Data Science journey
'''
print('Python')
```

    Python


It's a good idea to write comments before your function explaining what that function does.  
Commenting out lines can also help debug your code.

---
### String concepts
Strings can be stored in variables. Variables are just placeholders for some value inside our code.


```python
first_name = 'Pranav'
print(first_name)
```

    Pranav


#### Concatenate strings
You can combine strings with the `+` operator.


```python
first_name = 'John'
last_name = 'Doe'
print(first_name + last_name)
print('Hello, ' + first_name + ' ' + last_name)
```

    JohnDoe
    Hello, John Doe


#### Functions to modify strings
Below we have used functions to 
- convert a string to uppercase 
- convert a string to lowercase
- capitalize just the first word
- count all of the instances of a particular string.


```python
sentence = 'My name is John Doe'
print(sentence.upper())
print(sentence.lower())
print(sentence.capitalize())
print(sentence.count('o'))
```

    MY NAME IS JOHN DOE
    my name is john doe
    My name is john doe
    2


You can use the escape character (backslash) `\` to insert characters that are illegal in a string. An example of an illegal character is a single quote inside a string that is surrounded by single quotes.


```python
first_name = input('What\'s your first name? ')
last_name = input('What\'s is your last name? ')
print('Hello, ' + first_name.capitalize() + 
     ' ' + last_name.capitalize())
```

    What's your first name? JOHN
    What's is your last name? doe
    Hello, John Doe


#### Custom string formatting
To infuse things in strings dynamically, you can use string formatting. 


```python
first_name = 'John'
last_name = 'Doe'
```

There are two ways you can do this:
- formatting with `.format()` string method



```python
name = 'Hello, {} {}'.format(first_name, last_name)
print(name)
```

    Hello, John Doe


- formatting with string literals, called f-strings


```python
name = f'Hello, {first_name} {last_name}'
print(name)
```

    Hello, John Doe


---
### Working with numbers
Numbers can be stored in variables. Make sure the variables have meaningful names. We can pass those variables inside functions.


```python
pi = 3.14159
print(pi)
```

    3.14159


#### Math with Numbers
- `+` for addition
- `-` for subtraction
- `*` for multiplication
- `/` for division
- `**` for exponent


```python
num1 = 9
num2 = 5
print(num1 + num2)
print(num1 ** num2)
```

    14
    59049


#### Type Conversion
You cannot combine strings with numbers in Python. For e.g., executing the code below will result in an error:  



```python
days_in_Dec = 31
print(days_in_Dec + ' days in December')
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-25-07fae050a6ec> in <module>
          1 days_in_Dec = 31
    ----> 2 print(days_in_Dec + ' days in December')
    

    TypeError: unsupported operand type(s) for +: 'int' and 'str'


When displaying a string that contains numbers, you must convert the numbers into strings.


```python
days_in_Dec = 31 
print(str(days_in_Dec) + ' days in December')
```

    31 days in December


Numbers can be stored as strings. However, numbers stored as strings are treated as strings.


```python
num1 = '10'
num2 = '20'
print(num1 + num2)
```

    1020


Also, the input function always returns a string.


```python
num1 = input('Enter the first number: ')
num2 = input('Enter the second number:  ')
print(num1 + num2)
```

    Enter the first number: 35
    Enter the second number:  75
    3575


But here you can see that you have a number stored in a string. What if you want to treat it as a number and do math with it?  
You can do another data type conversion. The `int` function will convert it to a whole number, while the `float` function will convert it into a floating point number that might have decimal places.


```python
num1 = input('Enter the first number: ')
num2 = input('Enter the second number:  ')
print(int(num1) + int(num2))
print(float(num1) + float(num2))
```

    Enter the first number: 35
    Enter the second number:  75
    110
    110.0


---
### Working with Dates
We often need current date and time when logging errors and saving data. To get the current date and time, we need to use the `datetime` library.


```python
from datetime import datetime

# the now function returns a datetime object
current_date = datetime.now()

print('Today is: ' + str(current_date))
```

    Today is: 2021-06-05 11:49:47.183898


There are a whole bunch of functions you can use with `datetime` objects to manipulate dates.  
`timedelta` is used to define a period of time.


```python
from datetime import datetime, timedelta
today = datetime.now()
print('Today is: ' + str(today))

one_day = timedelta(days=1)
one_week = timedelta(weeks=1)
yesterday = today - one_day
past_week = today - one_week
print('Yesterday was: ' + str(yesterday))
print('One week ago was: ' + str(past_week))
```

    Today is: 2021-06-05 12:01:31.236665
    Yesterday was: 2021-06-04 12:01:31.236665
    One week ago was: 2021-05-29 12:01:31.236665


You can also control the format of the date displayed on the screen. You can request just the day, month, year, hour, minutes and even seconds.


```python
print('Day: ' + str(current_date.day))
print('Month: ' + str(current_date.month))
print('Year: ' + str(current_date.year))
```

    Day: 5
    Month: 6
    Year: 2021


Sometimes, you can receive a date as a string, and you might need to store it as a date. You'll need to convert it to a `datetime` object.


```python
birthday = input('When is your birthday (dd/mm/yyyy)? ')

# the strptime function allows you to mention the 
# format in which you'll be receiving the date
birthday_date = datetime.strptime(birthday, '%d/%m/%Y')
print('Birthday: ' + str(birthday_date))
```

    When is your birthday (dd/mm/yyyy)? 28/2/2000
    Birthday: 2000-02-28 00:00:00


So what date was it three days before you were born?


```python
birthday = input('When is your birthday (dd/mm/yyyy)? ')
birthday_date = datetime.strptime(birthday, '%d/%m/%Y')
print('Birthday: ' + str(birthday_date))
three_days = timedelta(days=3)
three_before = birthday_date - three_days
print('Date three days before birthday: ' + str(three_before))
```

    When is your birthday (dd/mm/yyyy)? 28/2/2000
    Birthday: 2000-02-28 00:00:00
    Date three days before birthday: 2000-02-25 00:00:00


---
### Error Handling 
*Error handling* is when you have a problem with your code that is running, and its not something that you're going to be able to predict when you push your code to production. For e.g., permissions issue, database change, server being down, etc. Basically things that happen in the wild, which you have no control over.  
*Debugging* is when you know that there's something wrong (a bug) with your code because you did something incorrectly, and you're going to have to go in and correct it.

The following tools we're going to talk about are concerned with error handling. There are three types of errors:
- syntax errors
- runtime errors
- logic errors



#### Syntax errors
With syntax errors, your code is not going to run at all. This type of error is easiest to track down.


```python
# this code won't run at all
x = 35
y = 75
if x == y
    print('x = y')
```


      File "<ipython-input-23-beb86a647368>", line 4
        if x == y
                 ^
    SyntaxError: invalid syntax



We're missing a colon after `y`, which is why we're getting the error above.

#### Runtime errors
With runtime errors, your code will run, but it will fail when it encounters the error.


```python
# this code will fail when run
x = 5
y = 0
print(x / y)
```


    ---------------------------------------------------------------------------

    ZeroDivisionError                         Traceback (most recent call last)

    <ipython-input-24-8064a0e4bb6b> in <module>
          2 x = 5
          3 y = 0
    ----> 4 print(x / y)
    

    ZeroDivisionError: division by zero


We're trying to divide by zero, which is not possible. Python tells you why you're getting the error and points towards the line which needs to be fixed. It's good practice to start from the line mentioned and work your way up to the error. Runtime errors can also be caused due to an error in the framework you're using, but the chances of that happening are extremely rare. Most probably, if you have a runtime error, it's because there's something wrong in your code.

#### Catching runtime errors
When a runtime error occurs, Python generates an exception during the execution and that can be handled, which avoids your program to interrupt.  
Exception handling:
- `try`: this block will test the excepted error to occur
- `except`: here you can handle the error
- `else`: if there is no exception, then this block will be executed
- `finally`: finally block always gets executed whether exception is generated or not

These tools are not used for finding bugs.


```python
x = 5
y = 0

try:
    print(x / y)
except ZeroDivisionError as e:
    print('Sorry, something went wrong')
except:
    print('Something really went wrong')
finally:
    print('This line always runs, on success or failure')
```

    Sorry, something went wrong
    This line always runs, on success or failure


#### Logic errors
Logic errors occur when the code compiles properly, doesn't give any syntax or runtime errors, but it doesn't give you the response you're looking for.


```python
# this code won't run at all
x = 10
y = 20
if x > y:
    print(str(x) + ' is less than ' + str(y))
```

In the code above, `x` is less than `y`; but the `if` statement includes `x > y`, instead of `x < y`.

When you're figuring out what went wrong with your code, just make sure that you reread your code. You can check the documentation and also search the internet on sites like StackOverflow and Medium.

---
### Handling Conditions
Your code might need the ability to take different actions based on different conditions. Below are the operations that you'll need for comparisons:
- `>`: greater than
- `<`: less than
- `>=`: greater than or equal to
- `<=`: less than or equal to
- `==`: is equal to
- `!=`: is not equal to

#### if statement
The `if` statement contains a logical expression using which the data is compared and a decision is made based on the result of the comparison.


```python
price = 250.0

if price >= 100.00:
    tax = 0.3
    print(tax)
```

    0.3


#### if - else statement
You can add a default action using `else`. An `else` statement contains the block of code that executes if the conditional expression in the `if` statement resolves to `0` or a `False` value.


```python
price = 50

if price >= 100.00:
    tax = 0.3
else:
    tax = 0
    
print(tax)
```

    0


Be careful when comparing strings. String comparisons are case sensitive.


```python
country = 'INDIA'
if country == 'india':
    print('Namaste')
else:
    print('Hello')
```

    Hello



```python
country = 'INDIA'
if country.lower() == 'india':
    print('Namaste')
else:
    print('Hello')
```

    Namaste


#### if - elif - else statement
You may need to check multiple conditions to determine the correct action. The `elif` statement allows you to check multiple expressions for `True` and execute a block of cide as soon as one of the conditions evaluates to `True`.


```python
# income tax percentage by state
state = input('Which state do you live in? ')

if state == 'Georgia':
    tax = 5.75
elif state == 'California':
    tax = 13.3
elif state == 'Texas' or state == 'Florida':
    tax = 0.0
else:
    tax = 4.0

print(tax)
```

    Which state do you live in? Georgia
    5.75


#### OR statements
| first condition | second condition | evaluation |  
|-----------------|------------------|------------|  
|True             |True              |True        |  
|True             |False             |True        |
|False            |True              |True        | 
|False            |False             |False       |

#### AND statements
| first condition | second condition | evaluation |  
|-----------------|------------------|------------|  
|True             |True              |True        |  
|True             |False             |False       |
|False            |True              |False       | 
|False            |False             |False       |

#### in operator
If you have a list of possible values to check, you can use the `in` operator.


```python
# income tax rates by state
state = input('Which state do you live in? ')

if state in ('Texas', 'Florida', 'Alaska',
             'Wyoming', 'South Dakota'):
    tax = 0.0
elif state == 'California':
    tax = 13.3
elif state == 'Georgia':
    tax = 5.75
else:
    tax = 4.0

print(tax)
```

    Which state do you live in? Alaska
    0.0


#### Nested if statement
There may be a situation when you want to check for another condition after a condition resolves to `True`. If an action depends on a combination of conditions, you can nest `if` statements.


```python
country = input("What country do you live in? ")

if country.lower() == 'canada':
    province = input("What province/state do you live in? ")
    if province in('Alberta', 'Nunavut','Yukon'):
        tax = 0.05
    elif province == 'Ontario':
        tax = 0.13
    else:
        tax = 0.15
else:
    tax = 0.0
print(tax)
```

    What country do you live in? Canada
    What province/state do you live in? Ontario
    0.13


Sometimes you can combine conditions with `and` instead of nested `if` statements.  
Let's assume that you're trying to calculate which students in a college have made the honor roll. The requirements for making the honor roll are a minimum 85% GPA and maintaining all your grades at at least 70%. 


```python
# convert strings into float
gpa = float(input('What\'s your GPA? '))
lowest_grade = float(input('What was your lowest grade? '))

if gpa >= 0.85 and lowest_grade >= 0.7:
    print('You made the honor roll')
else:
    print('You\'re really stupid')
```

    What's your GPA? 0.8
    What was your lowest grade? 0.75
    You're really stupid


If you have a very complicated `if` statement, rather than copying and pasting it in different parts of you code to do different things, we can remember what happened the last time we looked at the `if` statement with a Boolean variable.


```python
gpa = float(input('What\'s your GPA? '))
lowest_grade = float(input('What was your lowest grade? '))

if gpa >= 0.85 and lowest_grade >= 0.7:
    honor_roll = True
else:
    honor_roll = False
    
''' Somewhere later in your code if you need
to check if a student is on honor roll, all
you need to do is check the boolean variable
set earlier in the code'''
if honor_roll:  # True by default
    print('You made the honor roll')
```

    What's your GPA? 0.9
    What was your lowest grade? 0.87
    You made the honor roll


---
### Collections

####  Lists
Lists are a collection of items.


```python
# prepopulate a list
names = ['John', 'Will', 'Max']

# start with an empty list
scores = []
# add new item to the end
scores.append(90)
scores.append(91)

print(names)
print(scores)

# lists are zero-indexed
print(scores[1])
```

    ['John', 'Will', 'Max']
    [90, 91]
    91


You can get the number of items in a list using `len`.


```python
names = ['John', 'Will', 'Max']

# get the number of items using len
print(len(names))
```

    3


You can insert an item in a list using `insert`. This will insert the item at the specific index that you mention.


```python
# Bill will be inserted at index 0, i.e. the first item
names.insert(0, 'Bill')
print(names)
```

    ['Bill', 'John', 'Will', 'Max']


You can use `sort` to sort strings in alphabetical order. In case of numbers, it sorts them in the ascending order. Remember that using `sort` will modify the list!


```python
names.sort()
print(names)
```

    ['Bill', 'John', 'Max', 'Will']


You can retrieve a range within the list by indicating the start and end index; the end index being exclusive, i.e. it will not be included in the list.


```python
names = ['Amy', 'Susan', 'Jackie', 'Kylie', 'Ellen']

# start and end index
presenters = names[1:3]

# all names up to but not including index 3
hosts = names[:3]

# all names from 3 onwards, including index 3
judges = names[3:]

print(names)
print(presenters)
print(hosts)
print(judges)
```

    ['Amy', 'Susan', 'Jackie', 'Kylie', 'Ellen']
    ['Susan', 'Jackie']
    ['Amy', 'Susan', 'Jackie']
    ['Kylie', 'Ellen']


#### Arrays
Arrays are a collection of numbered data types. Unlike a list, in order for you to use an array, you have to create an array object by importing it from the `array` library.


```python
from array import array

# indicate the numerical type you'll use
scores = array('d') # d indicates a double
scores.append(80)
scores.append(81)
print(scores)
print(scores[0])
```

    array('d', [80.0, 81.0])
    80.0


So what's the difference between an array and a list?  
Arrays are only numerical data types and everything inside the array must be of the same data type. They can help add extra structure to your code.  
Lists can store anything you want, can store any data type, and can have mixed data types. They give more flexibility to your code.

#### Dictionaries
Dictionaries give you the ability to put together a group of items; but instead of using numeric indexes, you can use key-value pairs.


```python
person = {'first': 'John'}
person['last'] = 'Wick'
print(person)
print(person['first'])
```

    {'first': 'John', 'last': 'Wick'}
    John



```python
identity = {
    'Batman': 'Bruce Wayne',
    'Superman': 'Clark Kent',
    'Spiderman': 'Peter Parker',
    'Iron Man': 'Tony Stark'
}

print(identity)
```

    {'Batman': 'Bruce Wayne', 'Superman': 'Clark Kent', 'Spiderman': 'Peter Parker', 'Iron Man': 'Tony Stark'}


When to use a dictionary vs a list?
It depends on whether you want to name things and whether you want items to be in a guaranteed order.  
A dictionary will let you name key-value pairs but it does not guarantee you a specific order.  
A list does guarantee you a specific order since it has a zero-based index.

---

### Random Module

One way to introduce random numbers in your code is to use the `random` module.  
First you need to import the `random` module.


```python
import random

# generate a random whole number between 1 and 50
# inclusive of 1 and 50
random_integer = random.randint(1, 50)
print(random_integer)
```

    12



```python
# generate a random floating point number between 0.0 and 1.0
# exclusive of 1.0 
random_float = random.random()
print(random_float)

# generate a random floating point number between 0.0 and 5.0
print(random_float * 5)
```

    0.6343084141143187
    3.1715420705715935


There are so many more methods to the `random` module and you can check out the Python documentation to find out about all the things you can do with this module.

---
### Loops

Loops are a concept that is used when you need to have things happening over and over again.

#### for loops
`for` loops are used to loop through a collection. With a `for` loop, you can go through each item in a list and perform some action with each individual item in the list.

> `for item in list_of_items:
    # do something to each item`


```python
# go through the list of names
for name in ['John', 'Will', 'Max']:
    print(name)
```

    John
    Will
    Max



```python
wildcats = ['lion', 'tiger', 'puma', 'jaguar', 'cheetah', 'leopard']
for wildcat in wildcats:
    print(wildcat + ' is a wildcat.')
```

    lion is a wildcat.
    tiger is a wildcat.
    puma is a wildcat.
    jaguar is a wildcat.
    cheetah is a wildcat.
    leopard is a wildcat.


You can loop a particular number of times using `range`. `range` automatically creates a list of numbers for you. Remember that for the `range` function, the end index is exclusive.

> `for number in range(a, b):
    # do something
    print(number)`


```python
# end index is exclusive
for index in range(0, 5):
    print(index)
```

    0
    1
    2
    3
    4


If you want the range to increase by any other number, you can add a step to the function after the starting and ending indices.


```python
for index in range(0, 15, 3):
    print(index)
```

    0
    3
    6
    9
    12


#### while loop
`while` loops are used to loop with a condition. As long as something is `True`, the code will stay inside of the `while` loop i.e. the loop will continue going while the condition is true. You need to make sure that at some point you change the condition and it must result to `False`; otherwise the program will be stuck in an infinite loop, resulting in an error.

> `while something_is_true:
    # do something repeatedly`


```python
names = ['John', 'Will', 'Max']
index = 0
while index < len(names):
    print(names[index])
    # change the condition
    index += 1
    print(index)
```

    John
    1
    Will
    2
    Max
    3



```python
x = True
while x:
    print('This is an example of a while loop.')
    # change the condition
    x = False
    
while not x:
    print('This is another example of a while loop.')
    x = True
```

    This is an example of a while loop.
    This is another example of a while loop.


`for` loops are great when you want to iterate over something and you need to do something with each thing that you're iterating over. In cases like above, when you have a list, you almost always want use a `for` loop.  

`while` loops are useful when you don't care about the number in the sequence or about the item you're iterating through in a list, and you just simply want to carry out a functionality many times until a condition is met. You want to typically use a `while` loop when something is going to change automatically, e.g. when you need to read through a list of lines in a file, skip every alternate line, or if you're looking for something.  
`while` loops are more dangerous because they can lead to infinite loops if the condition is not met.

---
### Functions
A function is a block of organized, reusable code that is used to perform a single, related action. Function provide better modularity for your application and a high degree of code reusing, e.g. the `print()` function. You can create your own functions, called *user-defined functions*.  
Programming is all about copying and pasting code from one place to another. If you find yourself copying and pasting the exact same lines of code to more places in your program, you should probably move that into a function.  
Functions must be declared before the line of code where the function is called.  

**Defining Functions**  
> `def my_function():
    # do this
    # then do this
    # finally do this`
    
**Calling Functions**  
> `my_function()`



#### Functions with Inputs

The input to a function is something that can be passed over when we call the function.

> `def my_function(something):
    # do this with something
    # then do this
    # finally do this`
    
> `my_function(123)`

#### Functions with Outputs

The output keyword for a function is `return`. The `return` line must be the last line of the function. You can have multiple return keywords or even a blank return keyword in a function.


```python
def my_function():
    result = 5 * 4
    return result

my_function()
```




    20



When you call a function that has an output, the returned output is what replaces the function call, and the output can be stored as a variable.


```python
def my_function():
    return 5 * 8

output = my_function()
print(output)
```

    40


Imagine that you're trying to figure out why your program is taking a long time to run. So you write some print statements inside your code to tell you what time it is when the code is running, so you can see what time it is at different stages when your code is running.


```python
import datetime

# print timestamps to see how long
# sections take to run

first_name = 'John'
print('task completed')
print(datetime.datetime.now())
print()

for x in range(0, 10):
    print(x)
print('task completed')
print(datetime.datetime.now())
print()
```

    task completed
    2021-06-07 13:58:13.113387
    
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    task completed
    2021-06-07 13:58:13.115614
    


The above code can be rewritten using a function. You can define the function using `def` keyword, followed by the name of the function, and then a colon (`:`). Remember to use indentation which determines what code belongs to that function.


```python
# import datetime class from datetime library
from datetime import datetime

# print the current time
def print_time():
    print('task completed')
    # no need for the extra datetime prefix
    # since the class is imported above
    print(datetime.now()) 
    print()
    
first_name = 'John'
print_time()

for x in range(0, 10):
    print(x)
print_time()
```

    task completed
    2021-06-07 14:11:40.583077
    
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    task completed
    2021-06-07 14:11:40.589036
    


Sometimes when you copy/paste your code, we want to change some part of it. In the above example, what if you want to display a different message each time you run it. Say you want to display a specific message depending on the command you were running. This is where function parameters come in. *Parameters* or *arguments* are placed or defined within the parentheses of a function.


```python
from datetime import datetime

# print the current time and task name
def print_time(task_name):
    print(task_name)
    print(datetime.now())
    print()
    
first_name = 'John'
# pass in the task_name as a parameter
print_time('first name assigned')

for x in range(0, 10):
    print(x)
# pass in the task_name as a parameter
print_time('loop completed')
```

    first name assigned
    2021-06-07 16:07:57.077250
    
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    loop completed
    2021-06-07 16:07:57.079687
    


Let's take another example where the code looks different but we're using the same logic. Suppose you're interested in getting initials for a user ID after the user enters their name.


```python
first_name = input('Enter your first name: ')
# get only the first letter of input
first_name_initial = first_name[0:1]

last_name = input('Enter your last name: ')
last_name_initial = last_name[0:1]

print('Your initials are: ' + first_name_initial + last_name_initial)
```

    Enter your first name: John 
    Enter your last name: Wick
    Your initials are: JW


The above code can be written using a function.


```python
def get_initial(name):
    initial = name[0:1]
    # the return function returns a value
    return initial

first_name = input('Enter your first name: ')
first_name_initial = get_initial(first_name)

last_name = input('Enter your last name: ')
last_name_initial = get_initial(last_name)

# nested function in another call
print('Your initials are: ' + get_initial(first_name) + 
      get_initial(last_name))
```

    Enter your first name: john
    Enter your last name: wick
    Your initials are: jw


Functions can accept multiple parameters. In the above example, suppose you want to the user initials to only be uppercase for a user ID but lowercase for an email ID.


```python
def get_initial(name, force_uppercase=True): # default to True
    if force_uppercase:
        initial = name[0:1].upper()
    else:
        initial = name[0:1]
    return initial

first_name = input('Enter your first name: ')
first_name_initial = get_initial(first_name)

last_name = input('Enter your last name: ')
last_name_initial = get_initial(last_name, False)

print('Your initials are: ' + first_name_initial + last_name_initial)
```

    Enter your first name: john
    Enter your last name: wick
    Your initials are: Jw


When calling a function, you have to pass the parameters in the same order as when you defined the function. An exception to this is when you use named parameters, which offer better readability.  
`first_name_initial = get_initial(force_uppercase=True, name=first_name)`

Functions make the code more readable if you use good function names. They make the code less clunky. Always add comments to explain the purpose of your function.  
The main advantage of functions is that if you ever need to change your function code, you only need to change it in one place. You also reduce rework and the chance to introduce bugs when you change the code you copied.
