This blog post is a continuation of the previous post.

---
### Index

[List Comprehensions](#List-Comprehensions)  
[Dictionary Comprehensions](#Dictionary-Comprehensions)  
[Lambdas](#Lambdas)  
[Object-Oriented-Programming (OOP)](#Object-Oriented-Programming-(OOP))  
[Classes](#Classes)  
[Class Property](#Class-Property)  
[Inheritance](#Inheritance)  
[Managing the File System](#Managing-the-File-System)  

---
### List Comprehensions

List comprehension offers a shorter syntax when you want to create a new list based on the values of an existing list. Without list comprehension you will have to write a for statement with a conditional test inside.  
Say you have a list of numbers `[1, 2, 3]` and you want to create a new list where each number is increased by 1, then this is how you will do it using a `for` loop.


```python
numbers = [1, 2, 3]
new_list = []
for n in numbers:
    add_1 = n + 1
    new_list.append(add_1)
    
new_list
```




    [2, 3, 4]



Using list comprehensions, you can turn the above four lines of code into one. The syntax for list comprehensions is as follows:
> `new_list = [new_item for item in list]`  
Here `new_item` stands for the operation you want to perform or function you want to execute (i.e. `n+1`), `item` stands for the object you want to perform the operation on (i.e. `n`), and `list` stands for the list or collection you are iterating over (i.e. `numbers`).


```python
numbers = [1, 2, 3]
new_list = [n + 1 for n in numbers]
new_list
```




    [2, 3, 4]



You can use list comprehensions not only for lists but for any other sequences like tuples, strings, arrays, range, etc.  
Let's try to create a list of individual letters from a name. In this case, the `new_item` and `item` must be the same.


```python
name = 'John'
letters = [letter for letter in name]
letters
```




    ['J', 'o', 'h', 'n']



Let's try doubling the numbers obtained from a `range` function.


```python
new_range = [n * 2 for n in range(1, 5)]
new_range
```




    [2, 4, 6, 8]



You can also optionally add a condition to a list comprehension.
> `new_list = [new_item for item in list if test]`  
What this does is that it only performs the `new_item` function if the `test` is passed.  

From a list of cities, let's only pick the cities with the shortest names.


```python
cities = ['London', 'Toronto', 'Pune', 'Tokyo', 'New York', 'Zurich', 'Bern', 'Doha', 'Amsterdam']
short_names = [city for city in cities if len(city) < 5]
short_names
```




    ['Pune', 'Bern', 'Doha']



---
### Dictionary Comprehensions

Dictionary comprehensions help create a new dictionary from the values in an existing list or dictionary or any other type of collection.

Create a new dictionary with shortened syntax.
> `new_dict = {new_key:new_value for item in list}`

Create a new dictionary based on the values of an existing dictionary.
> `new_dict = {new_key:new_value for (key, value) in dict.items()}` 

You can also add an optional condition.
> `new_dict = {new_key:new_value for (key, value) in dict.items() if test}` 


Say you had a bunch of students and you wanted to assign them their exam scores. We'll generate a random score between 1 and 100.


```python
import random

names = ['Alex', 'Jason', 'Kelly', 'Jane', 'Jill', 'Joe']
student_scores ={student:random.randint(1, 100) for student in names}
student_scores
```




    {'Alex': 44, 'Jason': 77, 'Kelly': 30, 'Jane': 42, 'Jill': 29, 'Joe': 98}



Suppose that you need a score of 40 to pass. Let's create a dictionary of all students that have passed the exam.


```python
passed_students = {student:score for (student, score) in student_scores.items() if score >= 40}
passed_students
```




    {'Alex': 44, 'Jason': 77, 'Jane': 42, 'Joe': 98}



Let's try converting a bunch of temperatures in Celsius to Fahrenheit.


```python
weather_c = {
    "Monday": 12,
    "Tuesday": 14,
    "Wednesday": 15,
    "Thursday": 14,
    "Friday": 21,
    "Saturday": 22,
    "Sunday": 24,
}

weather_f = {day:(temp * 9/5 + 32) for (day, temp) in weather_c.items()}

weather_f
```




    {'Monday': 53.6,
     'Tuesday': 57.2,
     'Wednesday': 59.0,
     'Thursday': 57.2,
     'Friday': 69.8,
     'Saturday': 71.6,
     'Sunday': 75.2}



---
### Lambdas
A *lambda* function is an anonymous function. It can have any number of parameters but can only have one expression, which is evaluated and returned. You can use lambda functions wherever function objects are required.  

Suppose you want to sort some objects, where each object has a couple of properties. You need to tell `sort` how you want to sort the object. The `key` parameter allows you to pass in a function to call for each list element before it compares items for sorting.


```python
# sort people by name

# define function for sorting by name
def sorter(item):
    return item['name']

people  = [
    {'name': 'Tony', 'age': 45},
    {'name': 'Bruce', 'age': 40},
    {'name': 'Clark', 'age': 30},
    {'name': 'Peter', 'age': 18}
]

# sort people by name
people.sort(key=sorter)
print(people)
```

    [{'name': 'Bruce', 'age': 40}, {'name': 'Clark', 'age': 30}, {'name': 'Peter', 'age': 18}, {'name': 'Tony', 'age': 45}]


The `sorter` function is fairly small and it's not really doing a lot. When you've got a function like this that is just a single line of code, you don't necessarily have to declare a separate function. You can use a lambda function and it's implemented inline. 


```python
people  = [
    {'name': 'Tony', 'age': 45},
    {'name': 'Bruce', 'age': 40},
    {'name': 'Clark', 'age': 30},
    {'name': 'Peter', 'age': 18}
]

# lambda function to sort by name
people.sort(key=lambda item: item['name'])
print(people)

# lambda function to sort by length of name
people.sort(key=lambda item: len(item['name']))
print(people)
```

    [{'name': 'Bruce', 'age': 40}, {'name': 'Clark', 'age': 30}, {'name': 'Peter', 'age': 18}, {'name': 'Tony', 'age': 45}]
    [{'name': 'Tony', 'age': 45}, {'name': 'Bruce', 'age': 40}, {'name': 'Clark', 'age': 30}, {'name': 'Peter', 'age': 18}]


Before, we had the `sorter` function:
> `def sorted(parameter):
    return value`

This is how the lambda function syntax compares:
> `lambda parameter: value`

---
### Object Oriented Programming (OOP)

Let's say you want to start a restaurant, and in this restaurant you have 4 employees. A manager that manages a chef, a waiter, and a cleaner.  
Let us now consider the waiter. When considering the position of a waiter, we need to consider two things: what he has and what he does.

- What the waiter **has**:  
Can he hold a plate? What tables is he responsible for?

> `is_holding_plate = True`  
> `tables_responsible = [4, 5, 6]`

- What the waiter **does**:  
The waiter takes orders and takes payments from the customer.

> `def take_order(table, order): 
    #takes order to chef` 
    
> `def take_payment(amount): 
    # add money to restaurant`

What the object (the waiter) has are called **attributes**, and what the object does are called **methods**. As you can see, attributes are just variables associated with an object, whereas methods are just functions an object can do.

We can generate multiple versions of the same object. For e.g., we can have two waiters called John and Jenny. So now the object - waiter is a **class** and John and Jenny are its objects.

---
### Classes
A *class* is like an object constructor, a "blueprint" for creating objects. It allows you to define a data structure and its behavior, and store data. With classes, you can create reusable components, and group data and operations together.  
The attributes of a class are data members (class variables and instance variables) and methods, accessed via a dot(`.`) notation.
 
> Classes are nouns - what it is that you're describing  
Properties are adjectives - things that are true about the class  
Methods are verbs - things that the class can do

#### Creating a class
The names of the classes use Pascal casing.  
An *object* is a unique instance of a data structure that's defined by its class. An object comprises both, data members and methods.  
A *class variable* is a variable that is shared by all instances of a class. Class variables are defined within a class but outside any of the class's methods.  
An *instance* is an individual object of a certain class. For e.g., an object `obj` that belongs to a class `People` is an instance of the class.  
A *method* is nothing but a function that is defined within a class.  
An *instance variable* is a variable that is defined inside a method and belongs only to the current instance of a class.  
*Constructors* are used for instantiating an object. The task of constructors is to initialize (assign values) to the data members of the class when an object of class is created.  
In Python, the `__init__()` method is called the class constructor or initialization method, and it is always called when an object is created (constructed).  
The first parameter of `__init__` is `self`, which  is a reference to the current instance of the class, and is used to access variables that belong to the class. Note that it is not a keyword in Python and it does not have to be named `self`. You can call it whatever you like, but it has to be the first parameter of any function in the class. After that, we go ahead and set up any additional parameters that we may want.  
Then you define a field or property using `self.<parameter_name>`. So whenever we create an instance of the class, the `self` will refer to a different instance from which we are accessing class properties or methods.  
Then you can add any methods that you want, and the methods can involve any of the instantiated parameters, with the exception that the first argument to each method is `self`. Python adds the `self` argument to the list for you; you do not need to include it when you call the methods.


```python
# create class
class People():
    # class constructor
    def __init__(self, name):
        # field / property
        self.name = name
    # method
    def say_hello(self):
        print('Hello,  ' + self.name)
```

#### Using a class


```python
# set the name
person = People('Batman')
person.say_hello()

# change the name
person.name = 'Bruce'
person.say_hello()
```

    Hello,  Batman
    Hello,  Bruce


Here's another example:


```python
# class
class Laptop:
    # class constructor
    def __init__(self, company, model):
        # field
        self.company = company
        self.model = model

# creating instances for the class Laptop
laptop_one = Laptop('Apple', '13-inch Macbook Pro')
laptop_two = Laptop('Micrsoft', '15-inch Surface Book 3')

# printing the properties of the instances
print(f'Laptop 1: {laptop_one.company} {laptop_one.model}')
print(f'Laptop 2: {laptop_two.company} {laptop_two.model}')
```

    Laptop 1: Apple 13-inch Macbook Pro
    Laptop 2: Micrsoft 15-inch Surface Book 3


Let's focus on classes a bit more in detail. Suppose you want to create a class that keeps a count of the number of employees in the organization, and also displays their name and salary.


```python
class Employee:
    '''Common base class for all employees'''
    # class variable
    empCount = 0
    # can be accessed as Employee.empCount
    
    # class constructor
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1
        
    def displayCount(self):
        print('Total employees: %d' % Employee.empCount)
    
    def displayEmployee(self):
        print('Name: ', self.name, ', Salary: ', self.salary)
```

#### Creating instance objects
To create instances of a class, you call the class using the class name and pass in whatever arguments its `__init__` method accepts.


```python
# create first object of Employee class
emp1 = Employee('John', 40000)

# create second object of Employee class
emp2 = Employee('Jane', 60000)
```

#### Accessing attributes
You can access the object's attributes using the dot operator with the object. Class variable would be accessed using the class name.


```python
emp1.displayEmployee()
emp2.displayEmployee()
emp1.displayCount()
```

    Name:  John , Salary:  40000
    Name:  Jane , Salary:  60000
    Total employees: 2


One of the things that we have to be aware of when it comes to a field, is the fact that anybody is going to be able to access and update that. You can have some level of control over how somebody is able to use your class, which we'll discuss below.

#### Accessibility in Python
When you create a class, you might want to exert some level of control as to how you want people to use your class. For this, you need to learn about accessibility.  
Everything inside of Python is public. However, there are certain conventions for suggesting the accessibility.
- `_` (single underscore) means avoid that property or method unless you know exactly what you're doing. Maybe because there might be some changes in the property some time in the future.
- `__` (double underscore) means **do not use**.

#### Adding properties
In order to better control accessibility, you can use properties. Properties give us field style access but actually use methods behind the scenes.


```python
class People():
    def __init__(self, name):
        # constructor
        self.name = name # we're calling the property
    
    @property
    def name(self):
        print('In the getter')
        return self.__name
    
    @name.setter
    def name(self, value):
        print('In the setter')
        # validation here
        self.__name = value
```

#### Using a property


```python
person = People('Batman')
person.name = 'Bruce'
print(person.name)
```

    In the setter
    In the setter
    In the getter
    Bruce


We can see that "In the setter" is called twice because the first time we call it, is in the constructor.

### Class Property
The `property()` function, as the name suggests, is used to create a property of a class.

#### Class without getter and setters
Let's assume that we decide to make a class that stores the temperature in degrees Celsius. It would also implement a method to convert the temperature into degrees Fahrenheit. One way of doing this is as follows:


```python
class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature
    
    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
```

We can make objects out of this class and manipulate the `temperature` attribute as we wish.


```python
# basic method of setting and getting attributes in Python

# create a new object
human = Celsius()

# set the temperature
human.temperature = 37

# get the temperature attribute
print(human.temperature)

# get the to_fahrenheit method
print(human.to_fahrenheit())
```

    37
    98.60000000000001


Whenever we assign or retrieve any object attribute like `temperature`, Python searches it in the object's built-in `__dict__` dictionary attribute.


```python
human.__dict__
```




    {'temperature': 37}



Therefore, `man.temperature` internally becomes `man.__dict__['temperature']`.

#### Using getters and setters
Suppose we want to extend the usability of the `Celsius` class. We know that the temperature of any object cannot reach below -273.15 degrees Celsius (Absolute Zero in Thermodynamics). Let's update the code to implement this value restraint.  
An obvious solution to the above restriction will be to hide the attribute `temperature` (make it private) and define new getter and setter methods to manipulate it. This can be done as follows:


```python
# making getter and setter methods

class Celsius:
    def __init__(self, temperature=0):
        self.set_temperature(temperature)

    def to_fahrenheit(self):
        return (self.get_temperature() * 1.8) + 32
    
    # getter method
    def get_temperature(self):
        return self._temperature

    # setter method
    def set_temperature(self, value):
        if value < -273.15:
            raise ValueError('Temperature below -273.15 is not possible')
        self._temperature = value
```

The above method introduces two new `get_temperature()` and `set_temperature()` methods.  
Furthermore, `temperature` was replaced with `_temperature`. An underscore `_` at the beginning is used to denote private variables in Python.  
Let's use this implementation:


```python
# create a new object, 
# set_temperature() internally called by __init__
human = Celsius(37)

# get the temperature attribute via a getter
print(human.get_temperature())

# get the to_fahrenheit method, 
# get_temperature() called by the method itself
print(human.to_fahrenheit())
```

    37
    98.60000000000001



```python
# new contraint implementation
human.set_temperature(-300)

# get the to_fahrenheit method
print(human.to_fahrenheit())
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-9-2b566650c83f> in <module>
          1 # new contraint implementation
    ----> 2 human.set_temperature(-300)
          3 
          4 # get the to_fahrenheit method
          5 print(human.to_fahrenheit())


    <ipython-input-7-53b860660b5e> in set_temperature(self, value)
         15     def set_temperature(self, value):
         16         if value < -273.15:
    ---> 17             raise ValueError('Temperature below -273.15 is not possible')
         18         self._temperature = value


    ValueError: Temperature below -273.15 is not possible


This update successfully implemented the new restriction. We are no longer allowed to set the temperature below -273.15 degrees Celsius.
> **Note:** The private variables don't actually exist in Python. They are simply norms to be followed. The language itself doesn't apply any restrictions.


```python
human._temperature = -300
human.get_temperature()
```

However, the bigger problem with the above update is that all the programs that implemented our previous class have to modify their code from the `obj.temperature` to `obj.get_temperature()` and all expressions like `obj.temperature = val` to `obj.set_temperature(val)`.  
This refactoring can cause problems while dealing with hundreds of thousands of lines of codes. Basically, our new update was not backwards compatible. This is where `@property` comes to the rescue.

#### The property class
A Pythonic way to deal with the above problem is to use the `property` class. Here is how we can update our code:


```python
# using property class
class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature
    
    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
    
    # getter
    def get_temperature(self):
        print('Getting value...')
        return self._temperature
    
    # setter
    def set_temperature(self, value):
        print("Setting value...")
        if value < -273.15:
            raise ValueError('Temperature below -273.15 is not possible')
        self._temperature = value
        
    # creating a property object
    temperature = property(get_temperature, set_temperature)
```

We added a `print()` function inside `get_temperature()` to clearly observe that they are being executed. The last line of the code make a property object `temperature`. `property` attaches some code (`get_temperature()` and `set_temperature()`) to the member attribute accesses (`temperature`).  
Let's use this updated code:


```python
human = Celsius(37)

print(human.temperature)

print(human.to_fahrenheit())
```


```python
human.temperature = -300
```

As we can see, any code that retrieves the value of `temperature` will automatically call `get_temperature()` instead of a dictionary (`__dict__`) look-up. Similarly, any code that assigns a value to `temperature` will automatically call `set_temperature()`.  
We can even see that `set_temperature()` was called even when we created an object.  
Can you guess why?  
The reason is that when an object is created, the `__init__()` method gets called. This method has the line `self.temperature = temperature`. This expression automatically calls `set_temperature()`. Similarly, any access like `c.temperature` automatically calls `get_temperature()`. This is what property does. Here are a few more examples:


```python
shark = Celsius(20)
```


```python
shark.temperature
```


```python
shark.temperature = 20
```


```python
shark.to_fahrenheit()
```

By using `property`, we can see that no modification is required in the implementation of the value constraint. Thus, our implementation is backward compatible.
> **Note:** The actual temperature value is stored in the private `_temperature` variable. The `temperature` attribute is a property object which provides an interface to this private variable.

#### The  @property decorator
In Python, `property()` is a built-in function that creates and returns a `property` object. The syntax of this function is: 
> `property(fget=None, fset=None, fdel=None, doc=None)`  
> where,  
> - `fget` is a function to get value of the attribute  
> - `fset` is a function to set value of the attribute
> - `fdel` is a function to delete the attribute
> - `doc` is a string (like a comment)

These function arguments are optional. So, a property object can simply be created as follows:


```python
property()
```

To specify the arguments at a later point, a property object has three methods, `getter()`, `setter()`, and `deleter()`, to specify `fget`, `fset`, and `fdel` at a later point. This means, the line:
> `temperature = property(get_temperature, set_temperature)`

can be broken down as:
> `# make empty property`  
> `temperature = property()`  
> `# assign fget`  
> `temperature = temperature.getter(get_temperature)`  
> `# assign fset`  
> `temperature = temperature.setter(set_temperature)`

These two pieces of code are equivalent. The above construct can be implemented as decorators. We can even not define the names `get_temperature` and `set_temperature` as they are unnecessary and pollute the class namespace. For this, we reuse the `temperature` name while defining our getter and setter functions. Let's implement this as a decorator.


```python
# using @property decorator
class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature
        
    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
        
    @property
    def temperature(self):
        print('Getting value...')
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        print('Setting value...')
        if value < -273.15:
            raise ValueError('Temperature below -273.15 is not possible')
        self._temperature = value
        
# create an object
human = Celsius(37)

print(human.temperature)

print(human.to_fahrenheit())

coldest_thing = Celsius(-300)
```

The above implementation is not only simple and efficient, but it is also the recommended way to use `property`.

---
### Inheritance

Composition, with properties, creates a "has a" relationship:
> Student has a Class  
> DatabaseConnection has a ConnectionString  

Inheritance creates an "is a" relationship:
> Student is a Person  
> SqlConnection is a DatabaseConnection

*Inheritance* is the capability of one class to derive or inherit the properties from another class. It is the process in which we try to access the features of other classes without actually making the object of the parent class.  
*Parent class* is the class being inherited from, also called base class.  
*Child class* or derived class is the class that inherits from another class.


```python
class Person:
    def __init__(self, name):
        self.name = name
        
    def say_hello(self):
        print('Hello, ' + self.name)
        
class Student(Person):
    def __init__(self, name, school):
        super().__init__(name) # call the parent constructor
        self.school = school # field for Student
    def sing_school_song(self): # additional functionality
        print('Ode to ' + self.school)
        
student = Student('John', 'St. Paul')
student.say_hello() # comes from Person
student.sing_school_song() # comes from Student
```


```python
print(f'Is this a student? {isinstance(student, Student)}')
print(f'Is this a person? {isinstance(student, Person)}')
print(f'Is Student a Person? {issubclass(Student, Person)}')
```

What the above cell tells us is that:  
The variable `student` is an instance of the class `Student`.   
`student` is also an instance of the class `Person`.  
The class `Student` is a subclass (child class) of the superclass (parent class) `Person`.

We can also create a brand new definition of `say_hello()` in the `Student` class.


```python
class Person:
    def __init__(self, name):
        self.name = name
        
    def say_hello(self):
        print('Hello, ' + self.name)
        
class Student(Person):
    def __init__(self, name, school):
        super().__init__(name)
        self.school = school 
    def sing_school_song(self):
        print('Ode to ' + self.school)
    def say_hello(self):
        super().say_hello() # let parent class do some work
        print('You get good grades!') # add custom code
        
student = Student('John', 'St. Paul')
student.say_hello()
```

You need to use `super()` to call the function from the parent class. If you would not have used it here, the `say_hello()` function would not have printed "Hello, John".     
Let's see what happens when we just print `student`.


```python
print(student)
```

This just tells you that it's an object and gives you a memory address.


#### Overriding methods
You can override your parent class methods. One reason for overwriting parent's methods is because you may want special or different functionality in your subclass.


```python
class Student(Person):
    def __init__(self, name, school):
        super().__init__(name)
        self.school = school 
    def sing_school_song(self):
        print('Ode to ' + self.school)
    def say_hello(self):
        # super().say_hello()
        print('He gets good grades!') 
    def __str__(self):
        return f'{self.name} attends {self.school} school'
        
student = Student('John', 'St. Paul')
print(student)
student.say_hello()
```

Just like `__str__`, there are other methods you can use like `__init__(self[,args...])`, `__del__(self)`, `__repr__(self)`, `__cmp__(self,x)`. You can read more about them online.  
**Note:** You should not add functionality to your code unless you need it.

---
### Managing the File System
At some point, developers need to interact with the file system for reading or writing files, figuring out what directory you're in, etc. Let's look at some of the Python library systems we can use to work with the file system itself.
Earlier versions of Python used `os.path` i.e. operating system path, where you would call the operating system to check what path are you in. From Python 3.6 onwards, there is new library `pathlib`, which includes a class called `Path`; and this is a cleaner and faster way to access things like what directory you're in, what files are in that directory, etc.  
First, let's import the `Path` library.


```python
# grab the library
from pathlib import Path
```

If you want to retrieve what directory you are currently working in, you can use `Path.cwd`, where *cwd* stands for current working directory.


```python
# where am I?
cwd = Path.cwd()
print(cwd)
```

    /mnt/c/Users/prana/Data Science/Python


You might want to know the full path name, i.e. the name of the directory and the file name. The `joinpath` function builds for you the correct structure of the directory and file name together.


```python
# combine parts to create full path and file name
new_file = Path.joinpath(cwd, 'new_file.txt')
print(new_file)
```

    /mnt/c/Users/prana/Data Science/Python/new_file.txt


If you want to check if a file exists (maybe before reading the file), you can use the `exists` function.


```python
# will return False since I haven't actually created the file 
print(new_file.exists())
```

    False


#### Working with directories
When working with directories, we might need to check things like go into a directory, get a list of the contents of that directory, etc.


```python
# get the parent directory
parent = cwd.parent

# is this a directory?
print(parent.is_dir())

# is this a file?
print(parent.is_file())

# list child directories
for child in parent.iterdir():
    if child.is_dir():
        print(child)
```

    True
    False
    /mnt/c/Users/prana/Data Science/.ipynb_checkpoints
    /mnt/c/Users/prana/Data Science/Deep Learning Coursera TF
    /mnt/c/Users/prana/Data Science/Deep Learning with Pytorch
    /mnt/c/Users/prana/Data Science/FastAI
    /mnt/c/Users/prana/Data Science/fastpages_blog
    /mnt/c/Users/prana/Data Science/Hands-On Machine Learning
    /mnt/c/Users/prana/Data Science/ML Practice
    /mnt/c/Users/prana/Data Science/Programming PyTorch for DL
    /mnt/c/Users/prana/Data Science/PycharmProjects
    /mnt/c/Users/prana/Data Science/pyproj
    /mnt/c/Users/prana/Data Science/Python
    /mnt/c/Users/prana/Data Science/Python Basics
    /mnt/c/Users/prana/Data Science/Python for Data Analysis
    /mnt/c/Users/prana/Data Science/pyver
    /mnt/c/Users/prana/Data Science/source


#### Working with files
How do you find out information about the files you have?


```python
from pathlib import Path
cwd = Path.cwd()
demo_file = Path(Path.joinpath(cwd, 'demo.txt'))

# get the file name
print(demo_file.name)

# get the extension
print(demo_file.suffix)

# get the folder
print(demo_file.parent.name)

# get the size
print(demo_file.stat().st_size)
```

    demo.txt
    .txt
    Python
    702


For reading and writing files, most of the time you use a *file stream*.  
If you want to open a file, you create a *stream object* and you specify the name of the file you want to open, the mode, and the buffer size. The buffer size is mostly left at the default value. When using mode, by default Python assumes that you want to read the file.
> Modes:  
`r` - read (default)  
`w` - truncate and write (overwrites existing file)  
`a` - append if file exists (adds to existing file)  
`x` - write, fail if file exists   
`+` - updating (read/write)  
`t` - text (default)  
`b` - binary  

#### Reading from a file
If you create a stream and say `open()`, by default it's a text file and by default it's in read mode.


```python
stream = open('demo.txt')

# can we read?
print(stream.readable())

# read the first character
print(stream.read(1))

# read a line
print(stream.readline())

# close the stream
stream.close()
```

    True
    L
    orem ipsum dolor sit amet, consectetur adipiscing elit. Donec pharetra erat sed blandit dictum. Phasellus pharetra erat id molestie convallis. Phasellus rutrum nisl non turpis egestas pharetra. Nulla et nunc tristique arcu bibendum porttitor. In commodo vehicula porta. Phasellus neque lacus, placerat a risus id, feugiat suscipit ligula. Duis pretium, est eu elementum vulputate, elit enim sollicitudin mauris, sed mattis sem dolor in urna. Maecenas vestibulum ac elit suscipit ultrices. Donec hendrerit ante quis viverra aliquam. Cras vitae orci ac massa eleifend sollicitudin ut vel lorem. Nullam elementum elit id dolor tempus pharetra. Donec ac quam vel ex interdum pretium. Ut in vulputate leo. 


Note that when you `readline`, it continues from where the stream left off. So once it read the first character, the `readline` reads from the next character and does not re-read the line.

#### Writing to a file



```python
# write text
# "wt" stands for "write" "text"
stream = open('output.txt', 'wt') 

# write a single string
stream.write('H')

# write multiple strings
stream.writelines(['ello', ' ', 'world'])

# write a new line
# does not create new lines by default
stream.write('\n')

# you can pass list of strings
# create a list of strings
names = ['Jack ', 'John ', 'Jason ']

# write list of strings
stream.writelines(names)

# close the stream (and flush data)
stream.close()
```

You're not actually writing to the file. You're writing to the stream and then the stream goes to the file. You can reposition where you are in the stream using `seek`.


```python
stream = open('output.txt', 'wt')
stream.write('demo!')

# put the cursor back at the start
stream.seek(0)

# overwrite the file
# note that the 5th character (!) remains
stream.write('cool')
```




    4



Here, "demo" is overwritten by "cool", but the fifth character "!" remains because you haven't overwritten that character and the final text will be "cool!".  
The `flush` command flushes the data out of the stream to the file. If you do a flush and there's another piece of code out there where somebody else opens up the file, they will see the changes you made, for e.g. they will see the word "cool!" in the file. But this doesn't necessarily save the file.  
When you use `close`, it will flush anything that you haven't already flushed out and close the stream, and the file is saved permanently. So a lot of times, you can just `close` the file and not call `flush` explicitly.


```python
# write the data to file
stream.flush()

# flush and close the stream
stream.close() 
```

#### Cleaning with `with`

It is really important to close the stream after you're done with it because you can get error messages if you try to open something that is already open.  
You can use try/finally to avoid getting an error.


```python
try: 
    stream = open('output.txt', 'wt')
    stream.write('Lorem ipsum dolar')
finally:
    stream.close() # this is important!
```

A better alternative to the above code is using a `with` command. `with` will take care of closing for you. It says if the write is successful, close when we're done; if it gives an error, close when we're done.


```python
with open('output.txt', 'wt') as stream:
    stream.write('Lorem ipsum dolar')
```
