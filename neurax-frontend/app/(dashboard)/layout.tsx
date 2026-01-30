import { Header } from "@/components/common/Header"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container py-6">{children}</main>
    </div>
  )
}
