import os

# =========================================
# ELIMINAR ARCHIVOS TEMPORALES
# =========================================

def delete_file(filepath):

    if os.path.exists(filepath):
        os.remove(filepath)