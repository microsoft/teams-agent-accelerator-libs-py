from pydantic import BaseModel


class TaskConfig(BaseModel):
    task_name: str
    required_fields: list[str]


tasks_by_config = {
    "troubleshoot_device_issue": TaskConfig(
        task_name="troubleshoot_device_issue", required_fields=["OS", "Device Type", "Year"]
    ),
    "troubleshoot_connectivity_issue": TaskConfig(
        task_name="troubleshoot_connectivity_issue", required_fields=["OS", "Device Type", "Router Location"]
    ),
    "troubleshoot_access_issue": TaskConfig(
        task_name="troubleshoot_access_issue", required_fields=["OS", "Device Type", "Year"]
    ),
}
