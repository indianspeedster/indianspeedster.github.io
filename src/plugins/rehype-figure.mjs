/**
 * Turns a standalone image into a numbered <figure> with a caption.
 *
 *   ![alt text](/path.svg "Caption text")
 *
 * becomes
 *
 *   <figure>
 *     <img src="/path.svg" alt="alt text" loading="lazy" decoding="async">
 *     <figcaption><span class="fig-label">Figure 3.</span> Caption text</figcaption>
 *   </figure>
 *
 * Only images that are alone in their paragraph are converted, so inline
 * images (badges, icons mid-sentence) are left as they are. Numbering is
 * per-document and follows source order.
 *
 * Written without unist-util-visit so it pulls in no dependency of its own.
 */
export default function rehypeFigure() {
  return (tree) => {
    let count = 0;

    const isBlank = (node) =>
      node.type === 'text' && node.value.trim() === '';

    // A markdown image title is plain text, so `code spans` in a caption
    // arrive as literal backticks. Turn matched pairs into <code>.
    const withCodeSpans = (value) =>
      value
        .split(/`([^`]+)`/g)
        .map((part, i) =>
          i % 2
            ? {
                type: 'element',
                tagName: 'code',
                properties: {},
                children: [{ type: 'text', value: part }],
              }
            : { type: 'text', value: part }
        )
        .filter((node) => node.tagName || node.value !== '');

    const asLoneImage = (node) => {
      if (node.type !== 'element' || node.tagName !== 'p') return null;
      if (!Array.isArray(node.children)) return null;
      const kids = node.children.filter((c) => !isBlank(c));
      if (kids.length !== 1) return null;
      const [only] = kids;
      return only.type === 'element' && only.tagName === 'img' ? only : null;
    };

    const walk = (node) => {
      if (!Array.isArray(node.children)) return;

      node.children = node.children.map((child) => {
        walk(child);

        const img = asLoneImage(child);
        if (!img) return child;

        count += 1;

        // Markdown's image title carries the caption.
        const props = img.properties ?? (img.properties = {});
        const caption = props.title;
        delete props.title;
        props.loading = 'lazy';
        props.decoding = 'async';

        const figcaptionKids = [
          {
            type: 'element',
            tagName: 'span',
            properties: { className: ['fig-label'] },
            children: [{ type: 'text', value: `Figure ${count}.` }],
          },
        ];
        if (caption) {
          figcaptionKids.push(...withCodeSpans(` ${caption}`));
        }

        return {
          type: 'element',
          tagName: 'figure',
          properties: {},
          children: [
            img,
            {
              type: 'element',
              tagName: 'figcaption',
              properties: {},
              children: figcaptionKids,
            },
          ],
        };
      });
    };

    walk(tree);
  };
}
