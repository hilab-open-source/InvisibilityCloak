/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	darkMode: false,
	theme: {
		extend: {},
	},
	plugins: [require('@tailwindcss/typography'),],
	theme: {
		fontFamily: {
			'sans': ['Noto Sans']
		}
	}
}
