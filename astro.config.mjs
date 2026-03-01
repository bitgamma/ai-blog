// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	site: 'https://bitgamma.github.io',
	base: '/ai-blog/',
	trailingSlash: 'always',
	integrations: [mdx(), sitemap()],
  redirects: {
    '/': 'blog/'
  }
});
