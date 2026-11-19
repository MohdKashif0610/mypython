a=int(input("enter sub1 marks="))
b=int(input("enter sub2 marks="))
c=int(input("enter sub3 marks="))
d=int(input("enter sub4 marks="))
e=int(input("enter sub5 marks="))
total_mark=(a+b+c+d+e)
per=(total_mark*100)/500
if(per>=80):
    print("grade A")
elif(60<=per<80):
    print("grade B")
else:
    print("you have to workhard")

