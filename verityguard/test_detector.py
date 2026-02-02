# test_detector.py
from standalone_detector import StandaloneFakeNewsDetector

det = StandaloneFakeNewsDetector()

out = det.predict(
    text="Breaking news claim about a disaster",
    image_path=None,
    username="test_user"
)

print(out)
