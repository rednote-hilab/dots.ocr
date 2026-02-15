def construct_prompt(c1_text, c2_text):
    """
    Constructs the complete Prompt sent to Gemini (English Version).
    c1_text: Markdown text from Model 1
    c2_text: Markdown text from Model 2
    """
    
    prompt = f"""You are an expert in evaluating OCR content accuracy. Please compare the model outputs with the original image, focusing heavily on **content accuracy** while ignoring formatting and layout differences.

【Evaluation Focus - Focus ONLY on Content Accuracy】
1. **Text Accuracy**:
   - Typos: Character recognition errors (e.g., "test" recognized as "tost").
   - Omissions: Missing characters or words present in the original text.
   - Hallucinations: Adding characters that do not exist in the original text.

2. **Table Accuracy**:
   - Correctness of data and text within the table.
   - Completeness of cell content.
   - Correct row/column alignment.

3. **Formula Accuracy** (Evaluate based on):
   - **Correctness**: Are mathematical symbols, variables, and operators preserved accurately?
   - **Completeness**: Are all parts of the formula present without omission?
   - **Semantic Equivalence**: Does the extracted formula convey the exact same mathematical meaning?

【Tie Judgment Criteria - Important】
You must judge as a **tie** in the following cases:
- Text content is identical, differing only in Markdown formatting.
- Table data is identical, differing only in Markdown table syntax.
- Formula content is semantically equivalent, differing only in LaTeX representation.
- Both models correctly identified the core content; minor differences do not affect information retrieval.
- Both models share the same minor errors or are both perfect.
- **Image/Figure processing differs** (one extracts text, one gives bbox, one ignores it), but the main text is accurate.

【Items to Ignore - Do NOT factor into scoring】
- Markdown formatting differences (e.g., `# Header` vs `## Header`, `*` vs `-` for lists).
- Layout and typesetting differences (newlines, indentation, alignment).
- Recognition differences in non-body text like Headers, Footers, and Page Numbers.
- Text wrapping and paragraph segmentation nuances.
- Table border styles (e.g., `|---|---|` vs `|:--|--:|`).
- Different but equivalent LaTeX representations for formulas.
- **Image/Figure Processing Differences (ABSOLUTELY IGNORE)**: 
  - How the model parses image/figure regions is **completely excluded** from the scoring standard.
  - Whether it parses as a `figure` field, outputs bbox coordinates, extracts text inside the image, provides a caption, describes the image content, or **completely ignores/skips the image**, these are all considered equivalent.
  - Do NOT declare a winner based on image handling.

【Model 1 Output】:
```markdown
{c1_text} 
```

【Model 2 Output】:
```markdown
{c2_text}
```

【Evaluation Process】
1. Carefully compare the text content against the original image.
2. Identify errors, omissions, or additions in text recognition for both models.
3. Check the accuracy of table data.
4. Evaluate the correctness, completeness, and semantic equivalence of mathematical formulas.
5. **Ignore image regions**: Confirm that differences in image/figure parsing are not used for scoring.
6. Important: If the substance is the same and only the format differs, judge as a tie.
7. Only declare a winner if there is a significant difference in **content accuracy**.

【Examples of Ties】
- Model 1: "# Title", Model 2: "## Title" (Same content, different level).
- Model 1: "* Item", Model 2: "- Item" (Same content, different bullet).
- Formula: Model 1 "$x^2$", Model 2 "$x*x$" (Different LaTeX, same meaning).
- Table data is identical, but column alignment syntax differs.
- Identification is identical, but one model parsed the footer while the other didn't (Judge as Tie).
- **Image handling**: Model 1 outputs an image bbox, Model 2 outputs an image description, Model 3 ignores the image. As long as the main text is accurate, this is a **Tie**.

【Output Requirement】 Please strictly return the result in the following JSON format:

{{"winner": "tie", "reason": "Detailed explanation of the judgment, specifically noting the logic for a tie"}}

The value of "winner" must be one of:
- "1": Model 1 is clearly better in content accuracy.
- "2": Model 2 is clearly better in content accuracy.
- "tie": Both models perform equally in content accuracy (including cases of identical content but different formatting/image handling).

In the "reason" field, specifically explain:
- If a tie: Explain the consistency of the content and explicitly mention which formatting or image handling differences were ignored.
- If a winner: Specifically point out the accuracy differences (typos, missing words, table/formula errors).
- **Note**: It is better to judge a tie than to incorrectly determine a winner based on minor formatting or image parsing differences. **Content accuracy of the main text is the ONLY standard.**
"""
    
    return prompt