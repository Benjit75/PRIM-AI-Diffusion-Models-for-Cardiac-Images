"""

DIM_MULTS = (1, 2, 4)

MODEL_LOAD_PATH = "models/trained_models/"


model = Unet(
            dim=image_size,
            init_dim=None,
            out_dim=None,
            dim_mults=DIM_MULTS,
            channels=channels,
            with_time_emb=True,
            convnext_mult=2,
        ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=True, map_location=torch.device(DEVICE)))
    model.eval()

model2= ControlNET(
            dim=image_size,
            init_dim=None,
            out_dim=None,
            dim_mults=DIM_MULTS,
            channels=channels,
            with_time_emb=True,
            convnext_mult=2,
    )

Pipeline : 

1) Instancier un model model.py, un modèle controled UNET et un modèle controlNET
avec les bons weights

2) Copier les bons weights dans les bonnes catégories 
    -(les weights du UNET dans le controledUNET ne sont pas changés et comme 
    la classe hérite je peux juste l'appeller avec les weights du model.py
    qui sont les weights du UNET)
    - Les weights de controlNET doivent être copiés (instancier un model et prendre les weights
    des parties inférieures + les zero_convs sont initialisées à zero)

3) Save ces weights dans des fichiers que je pourrais LOAD ensuite (comme ça plus besoin de faire
l'opération de copiage à chaque fois)


Ensuite

1) Créer les Batchs avec les inputs et les hints
2) Instancier le modèle (controledUNET et controlNET)
3) Train le modèle avec les bons inputs, la bonne pipeline (apply_model, optimization, etc..) et
en updatant les bons weights (ne pas ajouter tout les weights à optimizer) (Train set - validation -
test set)
4) Récupérer des images pour tester sur le test set des segmentation, et aussi test des générations
pour voir si cela marche toujours bien (sachant que dans l'entraînement jsp si j'alterne ou pas entre
génération et segmentation)

"""
"""


## Store les weights du controlnet tel que je puisse l'instancier directement à
## partir du .txt ou .pt sans faire de copie


"""
