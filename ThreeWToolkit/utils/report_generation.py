from abc import ABC

class ReportGeneration(ABC):
    pass

    @staticmethod
    def generate_summary_report():
        pass

    @staticmethod
    def generate_feature_report():
        pass

    @staticmethod
    def save_report():
        pass

    @staticmethod
    def export_results_to_csv():
        pass

    @staticmethod
    def export_results_to_json():
        pass