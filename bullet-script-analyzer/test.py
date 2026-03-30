"""
Smoke test — verify the pipeline works end-to-end.

Run this BEFORE launching the Streamlit app to catch issues early:
    python test.py

This script calls analyze_script() with a minimal test input and prints
the structured result. If this works, the Streamlit app will work too.
"""

from analyzer.pipeline import analyze_script, get_available_model

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: Bullet Script Analyzer Pipeline")
    print("=" * 60)

    # Show which model will be used
    model = get_available_model()
    print(f"\n🤖 Using model: {model}")

    # Minimal test script — short enough for fast inference
    test_title = "Test Scene"
    test_script = "A man walks into a bar alone. He orders a drink. The bartender says, 'We've been expecting you.' The man looks up. 'How?' The bartender slides a photograph across the counter. It's the man — but twenty years younger, standing in front of this same bar."

    print(f"\n📝 Analyzing: '{test_title}'")
    print(f"   Script: {test_script[:80]}...")
    print("\n⏳ Running analysis (this may take 15-30 seconds on first run)...\n")

    try:
        result = analyze_script(test_title, test_script)
        print("✅ SUCCESS — Pydantic model parsed cleanly!\n")
        print(f"Title: {result.title}")
        print(f"Summary: {result.summary}")
        print(f"Dominant Emotions: {result.dominant_emotions}")
        print(f"Emotional Arc: {len(result.emotional_arc)} beats")
        for beat in result.emotional_arc:
            print(f"  - {beat.moment}: {beat.emotion} (intensity {beat.intensity}/10)")
        print(f"Engagement Score: {result.engagement_score}/10")
        print(f"Engagement Factors: {len(result.engagement_factors)}")
        for factor in result.engagement_factors:
            print(f"  - {factor.factor}: {factor.score}/10")
        print(f"Improvement Suggestions: {len(result.improvement_suggestions)}")
        for i, s in enumerate(result.improvement_suggestions, 1):
            print(f"  {i}. {s}")
        if result.cliffhanger:
            print(f"Cliffhanger: {result.cliffhanger.moment}")
        else:
            print("Cliffhanger: None detected")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise
