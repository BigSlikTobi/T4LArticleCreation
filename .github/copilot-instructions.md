## Bug/Error/Problem/Review Handling

When addressing bugs, errors, problems, or reviews, please follow this process:

1.  **Brainstorm Possibilities:**
    * Reflect on 5-7 different potential causes of the issue.

2.  **Distill to Likely Sources:**
    * Narrow down the possibilities to the 1-2 most likely root causes.

3.  **Validate with Logs:**
    * Add relevant logs to your code to validate your assumptions. This helps confirm the identified sources.

4.  **Implement Code Fix:**
    * Once the problem is confirmed, proceed with implementing the necessary code fix.

## Implementation Process

When implementing new features or changes, please follow this process:

1.  **Brainstorm Implementation Possibilities:**
    * Generate 5 different potential implementation approaches.

2.  **Evaluate and Select Best Approach:**
    * Evaluate each implementation based on:
        * Architecture: How well it fits with the existing system.
        * Implementation: Complexity and maintainability.
        * User Acceptance: Potential impact on user experience.
    * Select the best approach based on this evaluation.

3.  **Implement the Selected Approach:**
    * Implement the chosen approach.

4.  **Monitor Terminal for Issues:**
    * Pay close attention to the terminal output for any errors or warnings.

5.  **Address Terminal Issues:**
    * Resolve any reported issues before finalizing the implementation.

## Flutter UI Best Practices for Preventing Overflow

To ensure your Flutter UI is robust, responsive, and avoids overflow errors, follow these guidelines:

**1. Fundamental Layout Principles**

* **Use Flexible Widgets:** Employ `Flexible`, `Expanded`, and `FractionallySizedBox` within `Row` and `Column` to distribute space dynamically. Avoid using fixed sizes where possible, especially for children of rows and columns.

* **Layout Builders:** For complex, dynamic layouts, consider `LayoutBuilder` to get the parent's constraints and adjust child widgets accordingly.

* **SafeArea:** Wrap your top-level widget (often a `Scaffold`) with `SafeArea` to avoid clipping by the device's status bar, notch, or other system intrusions.

* **SingleChildScrollView:** Use `SingleChildScrollView` when content might exceed the screen size, allowing users to scroll. Combine this with `SafeArea` for best results.

* **IntrinsicWidth/IntrinsicHeight:** Use these with caution as they can be expensive, but they can be helpful for aligning widgets of varying sizes within a `Row` or `Column`.

**2. Overflow Prevention Strategies**

* **Check for Overflow:** Use the Flutter Inspector in your IDE (VS Code or Android Studio) to visually identify overflow errors during development. The inspector will highlight overflowing widgets in red.

* **Flexible Children in Rows/Columns:** Make sure children of `Row` and `Column` that need to adapt to available space are wrapped in `Flexible` or `Expanded`. `Expanded` is generally preferred as it forces the child to fill the available space, while `Flexible` allows more control over how the child fills the space using the `flex` factor.

* **Constrained Box:** Use `ConstrainedBox` to limit the maximum or minimum size of a widget, preventing it from expanding uncontrollably. For example, you might set a `maxWidth` on an image within a `Row`.

* **SizedBox with Constraints:** When you need a specific size, but also want to prevent overflow, use `SizedBox` in combination with `LayoutBuilder`.

* **Text Handling:**

    * **maxLines:** For `Text` widgets, set the `maxLines` property to limit the number of lines displayed. Use the `overflow` property to control how overflowed text is handled (e.g., `TextOverflow.ellipsis`).

    * **Flexible Text:** If a `Text` widget is in a `Row` or `Column`, wrap it in a `Flexible` or `Expanded` widget to allow it to take up the available space and wrap its text.

* **Image Handling:**

    * **BoxFit:** For `Image` widgets, use the `fit` property of the `Image` widget (e.g., `BoxFit.contain`, `BoxFit.cover`) to control how the image is scaled within its container.

    * **Constrained Size:** Explicitly set the `width` and `height` of `Image` widgets, or place them inside a `ConstrainedBox` to prevent them from exceeding their container's bounds.

* **Aspect Ratio:** Use the `AspectRatio` widget to maintain a specific aspect ratio, which can help prevent unexpected sizing issues.

**3. Best Practices for UI Development**

* **Consistent Spacing:** Use `Padding` and `SizedBox` consistently to create visual rhythm and spacing in your UI. Consider defining spacing constants (e.g., `const double kSmallPadding = 8.0;`) to ensure consistency.

* **Theme Usage:** Utilize Flutter's `Theme` to define colors, typography, and other visual properties. This promotes consistency and makes it easier to update your UI's appearance. Access theme properties using `Theme.of(context)`.

* **Widget Reusability:** Break down your UI into smaller, reusable widgets. This improves maintainability, testability, and reduces code duplication.

* **Performance Considerations:**

    * **Avoid Deep Nesting:** Excessive widget nesting can impact performance. Keep your widget tree as shallow as possible.

    * **Use const Where Possible:** Use the `const` keyword for widgets that don't change, allowing Flutter to optimize their rendering.

    * **Efficient List Building:** When building lists, use `ListView.builder` for large or dynamic lists to only build the widgets that are currently visible on screen.

* **Responsive Design:** Consider how your UI will adapt to different screen sizes and orientations. Use `MediaQuery` to get the screen size and orientation, and adjust your layout accordingly. Libraries like `responsive_builder` can simplify this.

* **Accessibility:** Make your UI accessible by providing appropriate labels, hints, and using semantic widgets.
