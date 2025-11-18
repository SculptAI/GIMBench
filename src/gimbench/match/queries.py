from gimkit import guide as g


user_profile = f"""I'm {g.person_name(name="pred")}. Hello, {g.single_word(name="obj")}!

My favorite hobby is {g.select(name="hobby", choices=["reading", "traveling", "cooking", "swimming"])}.

## Bio

{g(name="bio", desc="Four sentences.", regex=r"(?:[^.!?]+[.!?]){4}")}

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""

product_review = f"""
## Product Review

**Product Name:** {g.single_word(name="product_name")}
**Rating:** {g.select(name="rating", choices=["1", "2", "3", "4", "5"])}

**Review Title:** {g(name="review_title", desc="A catchy title for the review.")}

**Review:**
{g(name="review_body", desc="A detailed review of the product, at least 3 sentences.", regex=r"(?:[^.!?]+[.!?]){3,}")}
"""

email_body_regex = r"[\s\S]*\n\n[\s\S]*"
business_email = f"""
**To:** {g.e_mail(name="recipient_email")}
**From:** {g.e_mail(name="sender_email")}
**Subject:** {g(name="subject", desc="Subject of the email.")}

Dear {g.person_name(name="recipient_name")},

{g(name="email_body", desc="The body of the email, containing at least two paragraphs.", regex=email_body_regex)}

Best regards,
{g.person_name(name="sender_name")}
"""

story_generation = f"""
# A Short Story

Once upon a time, in a land called {g(name="land_name", desc="A fantasy land name")}, there lived a {g.select(name="protagonist_role", choices=["brave knight", "wise wizard", "cunning rogue", "curious princess"])} named {g.person_name()}.

The hero's journey began when they encountered a fearsome {g(name="antagonist", desc="An evil character or creature")}. The challenge was to {g(name="challenge", desc="The main goal of the story, at least 3 sentences.", regex=r"(?:[^.!?]+[.!?]){3,}")}.

After a long and arduous journey, {g.person_name()} finally {g.select(name="outcome", choices=["succeeded", "failed", "found a surprising twist"])}. The end.
"""

json_data = f"""
{{
    "name": "{g.person_name(name="name")}",
    "age": {g(name="age", regex=r"[1-9][0-9]?")},
    "isStudent": {g.select(name="is_student", choices=["true", "false"])},
    "courses": [
        "{g.single_word(name="course1")}",
        "{g.single_word(name="course2")}"
    ]
}}
"""

queries = {
    "User Profile": user_profile,
    "Product Review": product_review,
    "Business Email": business_email,
    "Story Generation": story_generation,
    "JSON Data": json_data,
}
