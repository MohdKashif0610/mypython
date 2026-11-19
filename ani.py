import turtle
import math
import random

# Screen setup
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Heart Curve with Colorful Stars")

# Turtle setup
t = turtle.Turtle()
t.speed(0)
t.hideturtle()
t.pensize(1)

# Colors
colors = [
    "red",
    "blue",
    "lime",
    "cyan",
    "magenta",
    "orange",
    "pink"
]

# Draw the heart
for i in range(120):

    t.penup()
    t.goto(0, 0)

    # Angle
    angle = i * (math.pi * 2) / 120

    # Heart curve
    x = 16 * (math.sin(angle) ** 3) * 12

    y = (
        13 * math.cos(angle)
        - 5 * math.cos(2 * angle)
        - 2 * math.cos(3 * angle)
        - math.cos(4 * angle)
    ) * 12

    # Random color
    c = random.choice(colors)
    t.color(c)

    # Draw line from center to heart
    t.pendown()
    t.goto(x, y)

    # Draw small star
    for _ in range(8):
        t.forward(6)
        t.backward(6)
        t.right(45)

turtle.done()