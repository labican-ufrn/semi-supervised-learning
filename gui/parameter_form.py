import customtkinter as ctk

class ParameterForm(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FlexCon - Configuração de Experimento")
        self.geometry("600x650")

if __name__ == "__main__":
    app = ParameterForm()
    app.mainloop()