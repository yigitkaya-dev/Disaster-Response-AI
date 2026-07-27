def generate_resource_recommendations(counts):
    """
    counts example:
    {
        "no-damage": 32,
        "minor-damage": 7,
        "major-damage": 12,
        "destroyed": 5
    }
    """

    total = sum(counts.values())

    if total == 0:
        return {
            "priority": "No data",
            "severe_percentage": 0,
            "search_rescue_teams": 0,
            "medical_units": 0,
            "shelter_units": 0,
            "inspection_teams": 0
        }

    major = counts.get("major-damage", 0)
    destroyed = counts.get("destroyed", 0)
    minor = counts.get("minor-damage", 0)

    severe = major + destroyed

    severe_percentage = (severe / total) * 100
    destroyed_percentage = (destroyed / total) * 100

    # Overall disaster priority
    if severe_percentage >= 50:
        priority = "Critical"
    elif severe_percentage >= 30:
        priority = "High"
    elif severe_percentage >= 15:
        priority = "Moderate"
    else:
        priority = "Low"

    # Simple resource allocation rules

    # 1 rescue team per 5 severely damaged buildings
    search_rescue_teams = max(
        1 if severe > 0 else 0,
        round(severe / 5)
    )

    # Extra medical focus when buildings are destroyed
    medical_units = max(
        1 if destroyed > 0 else 0,
        round((major + (destroyed * 2)) / 10)
    )

    # Assume major/destroyed buildings may create displaced residents
    shelter_units = round(severe / 4)

    # Minor + major structures need inspection
    inspection_teams = max(
        1 if (minor + major) > 0 else 0,
        round((minor + major) / 10)
    )

    return {
        "priority": priority,
        "severe_percentage": severe_percentage,
        "destroyed_percentage": destroyed_percentage,
        "search_rescue_teams": search_rescue_teams,
        "medical_units": medical_units,
        "shelter_units": shelter_units,
        "inspection_teams": inspection_teams
    }