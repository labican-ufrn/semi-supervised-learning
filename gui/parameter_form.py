import customtkinter as ctk

class ParameterForm(ctk.CTk):


    def __init__(self):
        super().__init__()

        # Configuração da janela
        ctk.set_appearance_mode("dark")

        self.title("FlexCon - Configuração de Experimento")
        self.geometry("700x820")
        self.resizable(False, False)
        self.configure(fg_color="#121820")

        # Painel principal
        self.frame = ctk.CTkFrame(
            self,
            width=520,
            corner_radius=15,
            fg_color="#1E293B"
        )
        self.frame.pack(pady=35, padx=40, fill="both", expand=True)

        # Título
        self.title_label = ctk.CTkLabel(
            self.frame,
            text="Configuração do Experimento",
            font=("Arial", 23, "bold"),
            text_color="#60A5FA"
        )
        self.title_label.pack(pady=(30, 5))

        self.subtitle_label = ctk.CTkLabel(
            self.frame,
            text="Defina os parâmetros do algoritmo",
            font=("Arial", 13),
            text_color="#94A3B8"
        )
        self.subtitle_label.pack(pady=(0, 25))

        # Percentual de dados rotulados
        self.label_percentage = ctk.CTkLabel(
            self.frame,
            text="Percentual de Dados Rotulados (0-100)",
            font=("Arial", 13, "bold"),
            text_color="#E2E8F0"
        )
        self.label_percentage.pack(anchor="w", padx=65, pady=(5, 5))

        self.entry_percentage = ctk.CTkEntry(
            self.frame,
            placeholder_text="Ex: 50",
            width=390,
            height=38,
            fg_color="#273548",
            border_color="#334155",
            text_color="#E2E8F0"
        )
        self.entry_percentage.pack(pady=(0, 15), anchor="w", padx=65)

        # Threshold
        self.label_threshold = ctk.CTkLabel(
            self.frame,
            text="Threshold (0-1)",
            font=("Arial", 13, "bold"),
            text_color="#E2E8F0"
        )
        self.label_threshold.pack(anchor="w", padx=65, pady=5)

        self.entry_threshold = ctk.CTkEntry(
            self.frame,
            placeholder_text="Ex: 0.5",
            width=390,
            height=38,
            fg_color="#273548",
            border_color="#334155",
            text_color="#E2E8F0"
        )
        self.entry_threshold.pack(pady=(0, 15), anchor="w", padx=65)

        # Change Rate (CR)
        self.label_cr = ctk.CTkLabel(
            self.frame,
            text="CR (0-1)",
            font=("Arial", 13, "bold"),
            text_color="#E2E8F0"
        )
        self.label_cr.pack(anchor="w", padx=65, pady=5)

        self.entry_cr = ctk.CTkEntry(
            self.frame,
            placeholder_text="Ex: 0.5",
            width=390,
            height=38,
            fg_color="#273548",
            border_color="#334155",
            text_color="#E2E8F0"
        )
        self.entry_cr.pack(pady=(0, 20), anchor="w", padx=65)

        # Classificadores
        self.label_classifiers = ctk.CTkLabel(
            self.frame,
            text="Selecione os classificadores",
            font=("Arial", 15, "bold"),
            text_color="#60A5FA"
        )
        self.label_classifiers.pack(anchor="w", padx=65, pady=(5, 15))

        self.classifiers = [
            "KNN",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "Logistic Regression",
            "Neural Network"
        ]

        self.classifiers_vars = {}

        # Organização em duas colunas
        self.classifier_frame = ctk.CTkFrame(
            self.frame,
            fg_color="transparent"
        )
        self.classifier_frame.pack(anchor="w", padx=65)

        for index, classifier in enumerate(self.classifiers):
            var = ctk.BooleanVar(value=False)

            checkbox = ctk.CTkCheckBox(
                self.classifier_frame,
                text=classifier,
                variable=var,
                font=("Arial", 12),
                checkbox_width=20,
                checkbox_height=20,
                fg_color="#2563EB",
                hover_color="#3B82F6",
                text_color="#E2E8F0"
            )

            row = index // 2
            column = index % 2

            checkbox.grid(
                row=row,
                column=column,
                padx=(0, 30),
                pady=10,
                sticky="w"
            )

            self.classifiers_vars[classifier] = var

if __name__ == "__main__":
    app = ParameterForm()
    app.mainloop()