import numpy as np

from src.utils.anatomy import to_anatomical_coords, from_anatomical_coords




def test_anatomical_registration():
    
    p = np.array([150, 400])
    prior_mask = np.zeros((512, 512), dtype=np.uint8)
    prior_centroid = np.array([256, 256])
    prior_nipple = np.array([256, 100])
    a = to_anatomical_coords(
        p, prior_mask, prior_centroid, prior_nipple
    )

    p_reconstructed = from_anatomical_coords(
        a, prior_mask, prior_centroid, prior_nipple
    )

    print(p)
    print(a)
    print(p_reconstructed)
    print("error:", np.linalg.norm(p - p_reconstructed))
    
    
if __name__ == "__main__":
    test_anatomical_registration()