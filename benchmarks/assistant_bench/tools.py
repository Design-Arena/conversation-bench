from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

book_flight_function = FunctionSchema(
    name="book_flight",
    description="Book a flight for a passenger. All required fields must be collected before calling.",
    properties={
        "origin": {
            "type": "string",
            "description": "Origin airport code (e.g., 'LAX').",
        },
        "destination": {
            "type": "string",
            "description": "Destination airport code (e.g., 'JFK').",
        },
        "date": {
            "type": "string",
            "description": "Flight date (e.g., '2025-02-03').",
        },
        "time": {
            "type": "string",
            "description": "Departure time in 24-hour format (e.g., '14:30').",
        },
        "passengers": {
            "type": "integer",
            "description": "Number of passengers.",
        },
        "passenger_name": {
            "type": "string",
            "description": "Full name of the primary passenger.",
        },
    },
    required=["origin", "destination", "date", "time", "passengers", "passenger_name"],
)

book_hotel_function = FunctionSchema(
    name="book_hotel",
    description="Book a hotel room. All required fields must be collected before calling.",
    properties={
        "city": {
            "type": "string",
            "description": "City name (e.g., 'New York').",
        },
        "check_in": {
            "type": "string",
            "description": "Check-in date (e.g., '2025-02-03').",
        },
        "check_out": {
            "type": "string",
            "description": "Check-out date (e.g., '2025-02-05').",
        },
        "hotel_name": {
            "type": "string",
            "description": "Name of the hotel.",
        },
        "room_type": {
            "type": "string",
            "description": "Room type (e.g., 'standard', 'suite', 'deluxe').",
        },
        "guest_name": {
            "type": "string",
            "description": "Full name of the guest.",
        },
    },
    required=["city", "check_in", "check_out", "hotel_name", "room_type", "guest_name"],
)

set_reminder_function = FunctionSchema(
    name="set_reminder",
    description="Set a reminder for a specific date and time.",
    properties={
        "date": {
            "type": "string",
            "description": "Reminder date (e.g., '2025-02-02').",
        },
        "time": {
            "type": "string",
            "description": "Reminder time in 24-hour format (e.g., '20:00').",
        },
        "message": {
            "type": "string",
            "description": "The reminder message.",
        },
    },
    required=["date", "time", "message"],
)

send_email_function = FunctionSchema(
    name="send_email",
    description="Send an email on behalf of the user.",
    properties={
        "to": {
            "type": "string",
            "description": "Recipient email address.",
        },
        "subject": {
            "type": "string",
            "description": "Email subject line.",
        },
        "body": {
            "type": "string",
            "description": "Email body text.",
        },
    },
    required=["to", "subject", "body"],
)

check_calendar_function = FunctionSchema(
    name="check_calendar",
    description="Check the user's calendar for a specific date.",
    properties={
        "date": {
            "type": "string",
            "description": "The date to check (e.g., '2025-02-03').",
        },
    },
    required=["date"],
)

add_calendar_event_function = FunctionSchema(
    name="add_calendar_event",
    description="Add a new event to the user's calendar.",
    properties={
        "date": {
            "type": "string",
            "description": "Event date (e.g., '2025-02-05').",
        },
        "time": {
            "type": "string",
            "description": "Event start time in 24-hour format (e.g., '10:00').",
        },
        "duration": {
            "type": "string",
            "description": "Event duration (e.g., '1 hour', '30 minutes').",
        },
        "title": {
            "type": "string",
            "description": "Event title.",
        },
        "notes": {
            "type": "string",
            "description": "Optional notes for the event.",
        },
    },
    required=["date", "time", "duration", "title"],
)

ToolsSchemaForTest = ToolsSchema(
    standard_tools=[
        end_session_function,
        book_flight_function,
        book_hotel_function,
        set_reminder_function,
        send_email_function,
        check_calendar_function,
        add_calendar_event_function,
    ]
)
