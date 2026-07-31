import os

BASE_PATH = "/data/wenxinma"
# 项目根目录下的 data 文件夹，用于存放 MVTec-AD / 私有数据集
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_PATH = {
    "Brain": f"{BASE_PATH}/data/MedAD/Brain_AD",
    "Liver": f"{BASE_PATH}/data/MedAD/Liver_AD",
    "Retina": f"{BASE_PATH}/data/MedAD/Retina_RESC_AD",
    "Colon_clinicDB": f"{BASE_PATH}/data/Colon/CVC-ClinicDB",
    "Colon_colonDB": f"{BASE_PATH}/data/Colon/CVC-ColonDB",
    "Colon_cvc300": f"{BASE_PATH}/data/Colon/CVC-300",
    "Colon_Kvasir": f"{BASE_PATH}/data/Colon/Kvasir",
    "BTAD": f"{BASE_PATH}/data/BTech_Dataset_transformed",
    "MPDD": f"{BASE_PATH}/data/MPDD",
    "MVTec": f"{LOCAL_DATA_PATH}/mvtec_ad",
    "VisA": f"{BASE_PATH}/data/VisA_20220922",
    # 私有数据集：根目录下需包含 object1, object2, ... 子文件夹
    "Private": LOCAL_DATA_PATH,
}

CLASS_NAMES = {
    "Brain": ["Brain"],
    "Liver": ["Liver"],
    "Retina": ["Retina"],
    "Colon_clinicDB": ["Colon_clinicDB"],
    "Colon_colonDB": ["Colon_colonDB"],
    "Colon_Kvasir": ["Kvasir"],
    "Colon_cvc300": ["CVC-300"],
    "MVTec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "transistor",
        "toothbrush",
        "wood",
        "zipper",
    ],
    "VisA": [
        "candle",
        "pcb3",
        "capsules",
        "pipe_fryum",
        "pcb4",
        "macaroni2",
        "pcb2",
        "chewinggum",
        "macaroni1",
        "cashew",
        "fryum",
        "pcb1",
    ],
    "MPDD": [
        "connector",
        "tubes",
        "metal_plate",
        "bracket_white",
        "bracket_brown",
        "bracket_black",
    ],
    "BTAD": ["01", "02", "03"],
    # 私有数据集：自动扫描 data/ 下的 object 文件夹。
    # 这里写死示例 object1/object2，实际使用时请替换为你的真实 object 名。
    # 也可运行 `python tools/gen_private_meta.py --scan-classes` 自动列出所有 object。
    "Private": [
        "object1",
        "object2",
    ],
}
DOMAINS = {
    "VisA": "Industrial",
    "BTAD": "Industrial",
    "MPDD": "Industrial",
    "MVTec": "Industrial",
    "Brain": "Medical",
    "Liver": "Medical",
    "Retina": "Medical",
    "Colon_clinicDB": "Medical",
    "Colon_colonDB": "Medical",
    "Colon_Kvasir": "Medical",
    "Colon_cvc300": "Medical",
    "Private": "Industrial",
}
REAL_NAMES = {
    "Brain": {"Brain": "scan"},
    "Liver": {"Liver": "scan"},
    "Retina": {"Retina": "scan"},
    "MVTec": {
        "bottle": "dark bottle",
        "cable": "top view of three cables",
        "capsule": "black and orange capsule",
        "carpet": "gray carpet",
        "grid": "metal or plastic mesh",
        "hazelnut": "single brown hazelnut",
        "leather": "brown leather",
        "metal_nut": "metal nut which has four notched edges",
        "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
        "screw": "screw",
        "tile": "speckled tile surface",
        "transistor": "a three-legged transistor placed vertically",
        "toothbrush": "toothbrush head",
        "wood": "wood surface",
        "zipper": "a black zipper",
    },
    "VisA": {
        "candle": "candle",
        "pcb3": "infrared sensor pcb module",
        "capsules": "capsules",
        "pipe_fryum": "pipe-shaped fryum",
        "pcb4": "battery charging pcb module",
        "macaroni2": "scattered yellow macaroni",
        "pcb2": "integrated circuits board",
        "chewinggum": "chewing gum",
        "macaroni1": "orange macaroni",
        "cashew": "cashew nut",
        "fryum": "wheel-shaped fryum snack",
        "pcb1": "dual ultrasonic distance sensor pcb module",
    },
    "Colon_clinicDB": {
        "Colon_clinicDB": "colon endoscopy image",
    },
    "Colon_colonDB": {
        "Colon_colonDB": "colon endoscopy image",
    },
    "Colon_cvc300": {"CVC-300": "colon endoscopy image"},
    "Colon_Kvasir": {"Kvasir": "colon endoscopy image"},
    "MPDD": {
        "connector": "metal clamps with black adjustment knobs",
        "tubes": "scattered metal objects",
        "metal_plate": "blue rectangular metal plate with a notch on one side",
        "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
        "bracket_brown": "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting holes along its arms",
        "bracket_black": "black ornamental metal bracket with spiral design attached to a rectangular frame",
    },
    "BTAD": {
        "01": "Bright concentric rings in neon yellow and blue tones against a dark blue background, resembling a stylized wave or energy field radiating outward.",
        "02": "vertical fabric lines in warm, dusty pink and beige tones",
        "03": "oval concentric circular rings in gradient shades of blue and white",
    },
    # 私有数据集：每个 object 的自然语言描述。
    # 若不知道怎么写，直接填 object 名本身也能跑（文本编码器会用它生成 prompt）。
    # 建议替换成更具语义的描述，例如 "red metal bracket"。
    "Private": {
        "object1": "object1",
        "object2": "object2",
    },
}
PROMPTS = {
    "prompt_normal": ["{}", "a {}", "the {}"],
    "prompt_abnormal": [
        "a damaged {}",
        "a broken {}",
        "a {} with flaw",
        "a {} with defect",
        "a {} with damage",
    ],
    "prompt_templates": [
        "{}.",
        "a photo of {}.",
    ],
}