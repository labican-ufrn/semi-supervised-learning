import customtkinter as ctk

class ParameterForm(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FlexCon - Configuração de Experimento")
        self.geometry("600x650")

        #labeled percentage
        self.label_percentage = ctk.CTkLabel(
            self,
            text="Percentual de Dados Rotulados (0-100):"
        )
        self.label_percentage.pack(pady=(20, 5))
        self.entry_percentage = ctk.CTkEntry(
            self,
            placeholder_text="Ex: 50"
        )
        self.entry_percentage.pack(pady=5)

        #threshold
        self.label_threshold = ctk.CTkLabel(
            self,
            text="Threshold (0-1):"
        )
        self.label_threshold.pack(pady=(20, 5))
        self.entry_threshold = ctk.CTkEntry(
            self,
            placeholder_text="Ex: 0.5"
        )
        self.entry_threshold.pack(pady=5)

        #cr
        self.label_cr = ctk.CTkLabel(
            self,
            text="CR (0-1):"
        )
        self.label_cr.pack(pady=(20,5))
        self.entry_cr = ctk.CTkEntry(
            self,
            placeholder_text="Ex: 0.5"
        )
        self.entry_cr.pack(pady=5)

if __name__ == "__main__":
    app = ParameterForm()
    app.mainloop()