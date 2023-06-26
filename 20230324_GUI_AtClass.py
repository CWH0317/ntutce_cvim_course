# https://visualtk.com/ A website to design your own GUI.
import tkinter as tk
import tkinter.font as tkFont

class App:
    def __init__(self, root):
        #setting title
        root.title("Temprature Converter")
        #setting window size
        width=600
        height=500
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        # Celsius text
        self.GLineEdit_540=tk.Entry(root)
        self.GLineEdit_540["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times',size=10)
        self.GLineEdit_540["font"] = ft
        self.GLineEdit_540["fg"] = "#333333"
        self.GLineEdit_540["justify"] = "center"
        #self.GLineEdit_540["text"] = "Entry"
        self.GLineEdit_540.place(x=220,y=80,width=290,height=64)
        # Convert Button
        self.GButton_179=tk.Button(root)
        self.GButton_179["bg"] = "#f0f0f0"
        ft = tkFont.Font(family='Times',size=22)
        self.GButton_179["font"] = ft
        self.GButton_179["fg"] = "#000000"
        self.GButton_179["justify"] = "center"
        self.GButton_179["text"] = "Convert"
        self.GButton_179.place(x=150,y=360,width=290,height=56)
        self.GButton_179["command"] = self.GButton_179_command
        # Farenheit text
        self.GLineEdit_604=tk.Entry(root)
        self.GLineEdit_604["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times',size=10)
        self.GLineEdit_604["font"] = ft
        self.GLineEdit_604["fg"] = "#333333"
        self.GLineEdit_604["justify"] = "center"
        #self.GLineEdit_604["text"] = "Entry"
        self.GLineEdit_604.place(x=220,y=190,width=290,height=62)

        self.GLabel_4=tk.Label(root)
        ft = tkFont.Font(family='Times',size=22)
        self.GLabel_4["font"] = ft
        self.GLabel_4["fg"] = "#333333"
        self.GLabel_4["justify"] = "center"
        self.GLabel_4["text"] = "Celsius"
        self.GLabel_4.place(x=40,y=80,width=113,height=63)

        self.GLabel_488=tk.Label(root)
        ft = tkFont.Font(family='Times',size=22)
        self.GLabel_488["font"] = ft
        self.GLabel_488["fg"] = "#333333"
        self.GLabel_488["justify"] = "center"
        self.GLabel_488["text"] = "Fahrenheit"
        self.GLabel_488.place(x=30,y=200,width=183,height=37)

    def GButton_179_command(self):
        if self.GLineEdit_604.get() != 0:
            self.GLineEdit_604.delete(0, tk.END)
        # Get the text in Celsius entry
        str_celdegree = self.GLineEdit_540.get()
        # Convert the text into a float
        celdegree = float(str_celdegree)
        # Convert the float from Celsius to Fahrenheit
        Fehdegree = (celdegree * (9 / 5)) + 32
        Fehdegree = round(Fehdegree, 2)
        # Convert the Fahrenheit
        str_Fehdegree = str(Fehdegree)
        self.GLineEdit_604.insert(0, str_Fehdegree)
        # Display the string to the 2nd entry


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
